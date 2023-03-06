import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch.nn.init import trunc_normal_
from torchvision.models.resnet import Bottleneck
from typing import List

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)  # 3 h w
    indices = indices[None]  # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw, w / 2.],
        [-sh, 0., h * offset + h / 2.],
        [0., 0., 1.]
    ]


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)  # (2,8,7,8,7,96):指把56*56的patch按照7*7的窗口划分
    # print(x.shape)  # (2,8,7,8,7,96)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size,
                                                            C)  # window的数量 H/7 * W/7 *batch
    # print(windows.shape)
    # windows=(128, 7, 7, 96)
    # 128 = batch_size * 8 * 8 = 128窗口的数量
    # 7 = window_size 窗口的大小尺寸，说明每个窗口包含49个patch
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            sigma: int,
            bev_height: int,
            bev_width: int,
            h_meters: int,
            w_meters: int,
            offset: int,
            decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)  # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))  # d h w

    def get_prior(self):
        return self.learned_features


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
#         super().__init__()
#
#         self.scale = dim_head ** -0.5
#
#         self.heads = heads  # 4  4  个头
#         self.dim_head = dim_head  # 32  应该是每个头的维度
#
#         self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
#         self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
#         self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
#
#         self.proj = nn.Linear(heads * dim_head, dim)
#         self.prenorm = norm(dim)
#         self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
#         self.postnorm = norm(dim)
#
#     def forward(self, q, k, v, skip=None):
#         """
#         q: (b n d H W)
#         k: (b n d h w)
#         v: (b n d h w)
#         """
#         _, _, _, H, W = q.shape
#
#         # Move feature dim to last for multi-head proj
#         q = rearrange(q, 'b n d H W -> b n (H W) d')  # 6，6，128，25，25到6，6，625，128
#         k = rearrange(k, 'b n d h w -> b n (h w) d')  # 6，6，128，56，120到6，6，6720，128
#         v = rearrange(v, 'b n d h w -> b (n h w) d')  # 6，6，128，56，120到6，40320，128
#
#         # Project with multiple heads
#         q = self.to_q(q)  # b (n H W) (heads dim_head)#四维 6，6，625，128
#         k = self.to_k(k)  # b (n h w) (heads dim_head)#都和上方一样
#         v = self.to_v(v)  # b (n h w) (heads dim_head)
#
#         # Group the head dim with batch dim
#         q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)  # 24，6，625，32
#         k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)  # 24，6，6270，32
#         v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
#
#         # Dot product attention along cameras
#         dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q,
#                                         k)  # 后面那个 shape=24，6，625，6720  dot 24,6,625,6720
#         dot = rearrange(dot, 'b n Q K -> b Q (n K)')  # dot 24,625,40320
#         att = dot.softmax(dim=-1)
#
#         # Combine values (image level features).
#         a = torch.einsum('b Q K, b K d -> b Q d', att, v)  # shape 24,625,32
#         a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)  # torch.Size([6, 625, 128])
#
#         # Combine multiple heads
#         z = self.proj(a)  # torch.Size([6, 625, 128])
#
#         # Optional skip connection
#         if skip is not None:
#             z = z + rearrange(skip, 'b d H W -> b (H W) d')
#
#         z = self.prenorm(z)
#         z = z + self.mlp(z)
#         z = self.postnorm(z)
#         z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)
#
#         return z


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, feature_map_size, num_heads, num_cams, qkv_bias=True,
                 attn_drop=0., proj_drop=0.,
                 norm=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_cams = num_cams
        self.num_heads = num_heads  # 多头的数量
        self.dim_head = dim // num_heads  # dim平均分给每个头
        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.minsize = min(feature_map_size[0], feature_map_size[1])
        self.pool = nn.AdaptiveAvgPool2d(self.minsize)
        # define a parameter table of relative position bias
        self.to_q = nn.Sequential(nn.Linear(dim, num_heads * self.dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(nn.Linear(dim, num_heads * self.dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(nn.Linear(dim, num_heads * self.dim_head, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_cams, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.add_qproj=nn.Linear(self.num_cams*dim,dim)
        self.softmax = nn.Softmax(dim=-1)
        self.prenorm = norm(self.dim * self.num_cams)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, q, k, v):
        """
#         q: (b n d H W)
#         k: (b n d h w)
#         v: (b n d h w)
#         """
        B, n, d, H, W = q.shape
        add_q=q.clone()
        add_q= rearrange(add_q, 'b n d H W -> b H W (n d)')
        add_q=self.add_qproj(add_q)
        #add_q.shape b, H W d
        add_q=rearrange(add_q, 'b H W d-> b (H W) d)')
        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')  # 6，6，128，25，25到6，6，625，128
        # Project with multiple heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q.shape b n (H W) d
        k = rearrange(k, 'b n h w d -> (b n) d h w ')
        v = rearrange(v, 'b n h w d -> (b n) d h w ')
        k = self.pool(k)
        v = self.pool(v)
        # k,v.shape b*n,d 14 14
        _, _, hk, wk = k.shape
        k = rearrange(k, '(b n) d h w -> b n h w d', b=B, n=n)
        v = rearrange(v, '(b n) d h w -> b n h w d', b=B, n=n)
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')
        #Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.num_heads, d=self.dim_head)  # 24，6，625，32
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.num_heads, d=self.dim_head)  # 24，6，6270，32
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.num_heads, d=self.dim_head)
        #q.shape b*nH  n (H W)   d/nH
        #k.shape b*nH  n (hk wk) d/nH
        #v.shape b*nH  (n hk wk) d/nH

        #Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q,k)  # 后面那个 shape=24，6，625，6720  dot 24,6,625,6720
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')  # dot 24,625,40320
        attn = self.softmax(dot)
        # attn (b*nH,n, H*W,hk*wk)→(b*nH H*W (n hk wk))
        attn = self.attn_drop(attn)
        #a .shape (b * nH  H*W  d/nH)
        a = torch.einsum('b Q K, b K d -> b Q d', attn, v)  # shape 24,625,32
        x = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)  # torch.Size([6, 625, 128])
        #x.shape b H*W d

        x = self.prenorm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x=x+add_q
        # x.shape B,H,W,d
        x = x + self.norm(self.mlp(x))
        x = rearrange(x, 'b (H W) d -> b d H W', H=H, W=W)
        return x


class CrossViewAttention(nn.Module):
    def __init__(
            self,
            feat_height: int,
            feat_width: int,
            feat_dim: int,
            dim: int,
            windows_size: int,
            image_height: int,
            image_width: int,
            qkv_bias: bool,
            heads: int = 4,
            dim_head: int = 32,
            stride=1,
            patch_size=3,
            no_image_features: bool = False,
            skip: bool = True,

    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))

        self.bev_embed = nn.Conv2d(2, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.img_embed = nn.Conv2d(4, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.cam_embed = nn.Conv2d(4, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.cross_attend = WindowAttention(dim, (feat_height, feat_width), heads, 6, qkv_bias)
        self.skip = skip

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(
            self,
            x: torch.FloatTensor,
            bev: BEVEmbedding,
            feature: torch.FloatTensor,
            I_inv: torch.FloatTensor,
            E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane  # b n 3 h w
        _, _, _, h, w = pixel.shape
        c = E_inv[..., -1:]  # b n 4 1，取出每一行最后一个数，组成一个4，1的矩阵（应该是平移参数）
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1最后这个[..., None] 相当于是多加了一维
        c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')  # pixel = img_plane 1 1 3 h w → 1 1 3 (h w)
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)
        d = E_inv @ cam  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat)  # (b n) d h w

        img_embed = d_embed - c_embed  # (b n) d h w d_embed 就是论文里面说的bev的position embedding （c） c_embed就是平移参数经过一个卷积
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d h w

        world = bev.grid[:2]  # 2 H W
        w_embed = self.bev_embed(world[None])  # 1 d H W
        bev_embed = w_embed - c_embed  # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)  # (b n) d h w
        else:
            key_flat = img_embed  # (b n) d h w

        val_flat = self.feature_linear(feature_flat)  # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w

        return self.cross_attend(query, key, val)


class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()
        i = 0
        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            if i == 0:
                window_size = 16
            else:
                window_size = 8
            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, window_size, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)  # b n c h w
        I_inv = batch['intrinsics'].inverse()  # b n 3 3
        E_inv = batch['extrinsics'].inverse()  # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()  # d H W  这边返回的x是一个可学习参数（其实就是论文里面的map embeddings)
        x = repeat(x, '... -> b ...', b=b)  # b d H W

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)

        return x
