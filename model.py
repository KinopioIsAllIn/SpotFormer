import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='face1',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'face0':
            self.num_node = 10
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = []
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'face1':
            self.num_node = 10
            # self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 2), (0, 1), (4, 3), (5, 4), (8, 6), (8, 7), (6, 9), (7, 9), (0, 3), (3, 8), (0, 8)]
            # neighbor_link = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (8, 7), (6, 9), (8, 9), (2, 3), (2, 7), (3, 7)]

            # self.edge = self_link + neighbor_link
            self.edge = neighbor_link
            self.center = 1
        elif layout == 'face2':
            self.num_node = 3
            # self_link = [(i, i) for i in range(self.num_node)]
            # neighbor_link = [(3, 0), (2, 1), (3, 2)]
            neighbor_link = [(1, 0), (1, 2), (0, 2)]
            # self.edge = self_link + neighbor_link
            self.edge = neighbor_link
            self.center = 1
        elif layout == 'face3':
            self.num_node = 1
            # self_link = [(i, i) for i in range(self.num_node)]
            # self.edge = self_link
            self.edge = []
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(1, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_undigraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(0, max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)  # 每一个节点有多少个邻居
    num_node = A.shape[0]  # 有多少个节点
    I = np.eye(num_node)
    Dn = np.zeros((num_node, num_node))  #
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    L = I - DAD
    return L

# The based unit of graph convolutional networks.

class ConvTemporalGraphical(nn.Module):

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
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        # N C T V
        x = self.conv(x)  # N, C, T, V

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)  # b, 1, c, t, v #
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # b, 1, c, t, v    1, v, v    ctv * vw = ctw

        return x.contiguous(), A

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

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
                 stride=1,
                 dilation=1,
                 dropout=0,
                 residual=True,
                 ifgcn=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        # padding = ((kernel_size[0] - 1) // 2 * dilation, 0)
        padding = (0, 0)
        self.ifgcn = ifgcn

        if ifgcn:
            self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1], t_kernel_size=kernel_size[0], t_stride=stride, t_dilation=dilation, t_padding=padding[0], bias=True)
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )

        else:
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    dilation=(dilation, 1),
                    padding=padding
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def maxpool2d(self, x, pool):
        if len(pool) == 0:
            return x
        p = pool[0]

        x0 = torch.max(x[:, :, :, p[0]:p[1]], dim=-1, keepdim=True)[0]
        for i in range(1, len(pool)):
            p = pool[i]
            x01 = torch.max(x[:, :, :, p[0]:p[1]], dim=-1, keepdim=True)[0]
            x0 = torch.cat([x0, x01], dim=-1)

        return x0

    def avgpool2d(self, x, pool):
        if len(pool) == 0:
            return x
        p = pool[0]

        x0 = torch.mean(x[:, :, :, p[0]:p[1]], dim=-1, keepdim=True)
        for i in range(1, len(pool)):
            p = pool[i]
            x01 = torch.mean(x[:, :, :, p[0]:p[1]], dim=-1, keepdim=True)
            x0 = torch.cat([x0, x01], dim=-1)

        return x0

    def forward(self, x, A, pool):
        res = self.residual(x)
        if self.ifgcn:
            x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        if len(pool) > 0:
            x = self.maxpool2d(x, pool)

        return self.relu(x), A


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, edge_importance_weighting):
        super().__init__()

        # load graph
        self.graph1 = Graph(layout='face1')
        self.graph2 = Graph(layout='face2')
        self.graph3 = Graph(layout='face3')

        A1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A1', A1)
        A2 = torch.tensor(self.graph2.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A2', A2)
        A3 = torch.tensor(self.graph3.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A3', A3)

        self.A = [A1, A2, A3, A3, A3, A3]

        pool1 = [[0, 3], [3, 6], [6, 10]]
        pool2 = [[0, 3]]
        pool3 = []

        self.pool = [pool1, pool2, pool3, pool3]

        kernel_size1 = (3, A1.size(0)) # 3
        kernel_size2 = (5, A1.size(0)) # 5
        kernel_size3 = (5, A1.size(0)) # 5
        kernel_size4 = (7, A1.size(0)) # 7

        self.data_bn = nn.BatchNorm1d(in_channels * A1.size(1))
        # kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        hidden_dim = 64
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_dim, kernel_size1, 1, residual=False, ifgcn=True),
            st_gcn(hidden_dim, hidden_dim, kernel_size2, 1, dilation=1, residual=False, ifgcn=True),
            st_gcn(hidden_dim, hidden_dim, kernel_size3, 1, dilation=1, residual=False, ifgcn=False),
            st_gcn(hidden_dim, hidden_dim, kernel_size4, 1, dilation=1, residual=False, ifgcn=False),
        ))

        # fcn for prediction
        self.fcn = nn.Conv2d(hidden_dim, num_class, kernel_size=(1, 1), padding=(0, 0), dilation=(1, 1))

        self._init_weight()

    def forward(self, x):
        # N, C, T, V = x.size()

        for gcn, A, pool in zip(self.st_gcn_networks, self.A, self.pool):
            A = A.to(x.device)
            x, _ = gcn(x, A, pool)

        x4contrastive = x

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1, 1)

        return x, x4contrastive.view(x.size(0), 1, -1)

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)



if __name__ == "__main__":
    pass
