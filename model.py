import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
import random



class GCNlayer(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            dilation=(1, 1),
            bias=bias)
        self.bn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            # nn.LayerNorm(out_channels),
            nn.Dropout(0.1, inplace=True),
        )


    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        # N T V C -> N C T V
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)  # N, C, T, V

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)  # b, 1, c, t, v #
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # b, 1, c, t, v    1, v, v    ctv * vw = ctw

        # x = x.contiguous().permute(0, 2, 3, 1).contiguous()
        # x = self.bn(x)
        x = self.bn(x.contiguous()).permute(0, 2, 3, 1).contiguous()

        return x, A

# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        n, a, b, c = x.shape
        x = x.reshape(-1, b, c)

        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()

        x = self.net(x)

        x = self.net2(x)
        return x.reshape(n, a, b, c)

        # return self.net(x).reshape(n, a, b, c)


class GraphAttention(nn.Module):
    def __init__(self, depth, dim, heads=8, dim_head=64, dropout=0., local=False):
        super().__init__()
        self.local = local
        self.depth = depth

        if local:
            self.mask = np.zeros((10, 10))
            if depth == 0:
                # self.mask = np.zeros((10, 10))
                self_link = [(i, i) for i in range(10)]
                self.connection = [[0, 1], [0, 2], [2, 1], [3, 4], [3, 5], [4, 5], [6, 7], [6, 8], [6, 9], [8, 7], [9, 7], [8, 9]]
            elif depth == 1:
                self.connection = [[0, 3], [3, 8], [0, 8]]
                self_link = [[0, 0], [3, 3], [8, 8]]
            for i in self.connection:
                self.mask[i[0]][i[1]] = 1.0
                self.mask[i[1]][i[0]] = 1.0
            for i in self_link:
                self.mask[i[0]][i[1]] = 1.0
            self.mask = torch.tensor(self.mask, dtype=torch.float32, requires_grad=True)

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.BatchNorm1d(dim)
        # self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        n, t, vn, c = x.shape
        if self.local and self.depth == 1:
            residual = x
        x = x.reshape(-1, vn, c)

        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.local:
            dots = dots * self.mask.to(x.device)


        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if self.local and self.depth == 1:
            out = self.to_out(out).reshape(n, t, vn, c)
            residual[:, :, 0:3] = residual[:, :, 0:3] + out[:, :, 0:1]
            residual[:, :, 3:6] = residual[:, :, 3:6] + out[:, :, 3:4]
            residual[:, :, 6:10] = residual[:, :, 6:10] + out[:, :, 8:9]
            return residual
        else:
            out = self.to_out(out).reshape(n, t, vn, c)
            return out


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.BatchNorm1d(dim)
        # self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        n, vn, t, c = x.shape
        x = x.reshape(-1, t, c)

        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out).reshape(n, vn, t, c)


class GraphDownsample(nn.Module):
    def __init__(self, depth, dim):
        super().__init__()

        self.depth = depth
        if depth == 0:
            self.pool = [[0, 3], [3, 6], [6, 10]]
        elif depth == 1:
            self.pool = [[0, 3]]  # [0, 5]
        else:
            self.pool = []

        self.norm = nn.BatchNorm2d(dim)
        # self.norm = nn.LayerNorm(dim)

    def maxpool2d(self, x, pool):
        if len(pool) == 0:
            return x

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        p = pool[0]
        x0 = torch.max(x[:, p[0]:p[1], :, :], dim=1, keepdim=True)[0]
        for i in range(1, len(pool)):
            p = pool[i]
            x01 = torch.max(x[:, p[0]:p[1], :, :], dim=1, keepdim=True)[0]
            x0 = torch.cat([x0, x01], dim=1)
        return x0

    def forward(self, x):
        if len(self.pool) > 0:
            x = self.maxpool2d(x, self.pool)
        return x


class TemporalDownsample(nn.Module):
    def __init__(self, depth, dim, padding, with_norm=True, dropout=0.):
        super().__init__()
        self.depth = depth

        if with_norm:
            self.norm = nn.BatchNorm1d(dim)
        else:
            self.norm = nn.Identity()
        self.downCNN = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=padding, bias=not with_norm)
    def forward(self, x):
        n, v, t, c = x.shape
        x = x.reshape(-1, t, c)  # .permute([0, 2, 1]).contiguous()  # -1, c, t

        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()

        x = x.permute([0, 2, 1]).contiguous()
        x = self.downCNN(x)

        _, c, t = x.shape
        return x.permute([0, 2, 1]).contiguous().reshape(n, v, t, c)


def graph_downsample(x):
    pool = [[0, 3], [3, 6], [6, 10]]

    p = pool[0]
    x0 = torch.max(x[:, p[0]:p[1], :], dim=1, keepdim=True)[0]
    for i in range(1, len(pool)):
        p = pool[i]
        x01 = torch.max(x[:, p[0]:p[1], :], dim=1, keepdim=True)[0]
        x0 = torch.cat([x0, x01], dim=1)
    return x0


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.temporal_downsample_layers = nn.ModuleList([])

        # list every layer
        self.gattn1 = GraphAttention(0, dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.gff1 = FeedForward(dim, mlp_dim, dropout=dropout)
        self.gattn2 = GraphAttention(1, dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.gff2 = FeedForward(dim, mlp_dim, dropout=dropout)

        self.gdown1 = GraphDownsample(0, dim)
        self.gdown2 = GraphDownsample(1, dim)

        self.padding = [0, 0, 1, 1] # [1, 1, 1, 0]   01111
        for i in range(len(self.padding)):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
            self.temporal_downsample_layers.append(nn.ModuleList([
                TemporalDownsample(i, dim, self.padding[i])
            ]))

    def forward(self, x):
        # n, t1, v1, c = x.shape
        x1 = x
        x1 = self.gattn1(x1) + x1
        x1 = self.gff1(x1) + x1

        x1 = x1.permute(0, 2, 1, 3).contiguous()  # .reshape(-1, t, c)  scale1   n v t c

        for layer in self.layers[0]:
            x1 = layer(x1) + x1

        x2 = self.gdown1(x1)
        for layer in self.temporal_downsample_layers[0]:
            x2 = layer(x2)

        # n, v2, t1, c = x2.shape
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        x2 = self.gattn2(x2) + x2
        x2 = self.gff2(x2) + x2  # scale1
        x2 = x2.permute(0, 2, 1, 3).contiguous()  # .reshape(-1, t, c)  # scale2  n v t c

        for layer in self.layers[1]:
            x2 = layer(x2) + x2
        x3 = self.gdown2(x2)  # scale3
        for layer in self.temporal_downsample_layers[1]:
            x3 = layer(x3)

        for layer in self.layers[2]:
            x3 = layer(x3) + x3
        for layer in self.temporal_downsample_layers[2]:
            x3 = layer(x3)

        for layer in self.layers[3]:
            x3 = layer(x3) + x3
        for layer in self.temporal_downsample_layers[3]:
            x3 = layer(x3)

        # for layer in self.layers[4]:
        #     x3 = layer(x3) + x3
        # for layer in self.temporal_downsample_layers[4]:
        #     x3 = layer(x3)

        return x3


class MSSTGT(nn.Module):
    def __init__(self, *, seq_len, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # assert (seq_len % patch_size) == 0

        self.to_patch_embedding = nn.Linear(2, dim)

        self.norm = nn.BatchNorm1d(dim)

        # pos embedding with parameters
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, dim))

        self.spatial_pos_embedding = nn.Parameter(torch.randn(3, dim))  # 2. different when symmetric or asymmetric


        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, num_classes)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute([0, 3, 2, 1]).contiguous().reshape(N, V*T, C)
        x = self.to_patch_embedding(x)  # 每个图节点需要patch embedding
        _, _, C = x.size()

        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous()

        x = x.reshape(N, V, T, C).reshape(N*V, T, C)
        self.temporal_pos_embedding = self.temporal_pos_embedding.to(x.device)
        x += self.temporal_pos_embedding[:, :T]  # temporal position embedding

        self.spatial_pos_embedding = self.spatial_pos_embedding.to(x.device)
        x = x.reshape(N, V, T, C).permute(0, 2, 1, 3).contiguous().reshape(N*T, V, C)
        # x += self.spatial_pos_embedding
        x[:, 0:3] += self.spatial_pos_embedding[0]
        x[:, 3:6] += self.spatial_pos_embedding[1]
        x[:, 6:10] += self.spatial_pos_embedding[2]

        x = self.dropout(x)
        x = x.reshape(N, T, V, C)
        # x = x.reshape(N, V, T, C).permute(0, 2, 1, 3).contiguous()

        x = self.transformer(x)[:, 0]  # 去掉一维 b, 1, 1, c -> b, 1, c

        return self.mlp_head(x).permute([0, 2, 1]).contiguous(), x, self.spatial_pos_embedding
