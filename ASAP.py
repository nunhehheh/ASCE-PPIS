from typing import Tuple, Union
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import Linear
from torch import Tensor
from torch_scatter import scatter_add, scatter_max
import torch_geometric
from torch_geometric.nn import (
    Aggregation,
    MeanAggregation,
    MessagePassing,
    SumAggregation,
)
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm


# torch.set_num_threads(1)


def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for the pooled graph using the formula S^{T}AS"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    # index_E, value_E = spspmm(index_St.cpu(), value_St.cpu(), index_B.cpu(), value_B.cpu(), kN, N, kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    # return index_E.to(device), value_E.to(device)
    return index_E, value_E


def graph_connectivity(device, perm, edge_index, edge_weight, score, ratio, batch, N):
    r"""graph_connectivity: is a function which internally calls StAS func to maintain graph connectivity"""

    kN = perm.size(0)
    perm2 = perm.view(-1, 1)

    # mask contains bool mask of edges which originate from perm (selected) nodes
    mask = (edge_index[0] == perm2).sum(0, dtype=torch.bool)

    # create the S
    S0 = edge_index[1][mask].view(1, -1)
    S1 = edge_index[0][mask].view(1, -1)
    index_S = torch.cat([S0, S1], dim=0)
    value_S = score[mask].detach().squeeze()

    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long,device=device)
    n_idx[perm] = torch.arange(perm.size(0), device=device)

    index_S[1] = n_idx[index_S[1]]

    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()

    fill_value = 1
    index_E, value_E = StAS(index_A, value_A, index_S, value_S, device, N, kN)
    index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
    index_E, value_E = add_remaining_self_loops(edge_index=index_E, edge_attr=value_E,
                                                fill_value=fill_value, num_nodes=kN)

    return index_E, value_E


class ASAP_Pooling(torch.nn.Module):

    def __init__(self, in_channels, ratio=0.3, dropout_att=0, negative_slope=0.2,clean=False):
        super(ASAP_Pooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = dropout_att
        self.lin_q = Linear(in_channels, in_channels)
        self.gat_att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)  # gnn_score: uses LEConv to find cluster fitness scores
        self.gnn_intra_cluster = EGNNLayer(
            node_features=self.in_channels,
            edge_features=0,
            hidden_features=self.in_channels,
            out_features=self.in_channels,
            act=nn.SiLU(),
            dropout=0.0,
            node_aggr=SumAggregation,
            cord_aggr=MeanAggregation,
            residual=True,
            update_coords=True,
            norm_coords=True,
            norm_coors_scale_init=1e-2,
            norm_feats=True,
            initialization_gain=1,
            attention=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.gat_att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index,pos, edge_weight=None, batch=None,clean=False):
        #edge_index = torch.stack(edge_index, dim=0)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # NxF
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Add Self Loops
        fill_value = 1
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index=edge_index, edge_attr=edge_weight,
                                                           fill_value=fill_value, num_nodes=num_nodes.sum())

        N = x.size(0)  # total num of nodes in batch

        # ExF
        x_pool,_ = self.gnn_intra_cluster(x,pos,edge_index)

        x_pool_j = x_pool[edge_index[1]]
        x_j = x[edge_index[1]]

        # ---Master query formation---
        # NxF
        X_q, _ = scatter_max(x_pool_j, edge_index[0], dim=0)
        # NxF
        M_q = self.lin_q(X_q)
        # ExF
        M_q = M_q[edge_index[0].tolist()]

        score = self.gat_att(torch.cat((M_q, x_pool_j), dim=-1))
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[0], num_nodes=num_nodes.sum())

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout_att, training=self.training)
        # ExF
        v_j = x_j * score.view(-1, 1)
        # ---Aggregation---
        # NxF
        out = scatter_add(v_j, edge_index[0], dim=0)

        # ---Cluster Selection
        # Nx1
        fitness = torch.sigmoid(self.gnn_score(x=out, edge_index=edge_index)).view(-1)
        return fitness

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__, self.in_channels, self.ratio)



class LEConv(torch.nn.Module):
    r"""Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(LEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        num_nodes = x.shape[0]
        h = torch.matmul(x, self.weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=x.dtype,
                                     device=edge_index.device)
        edge_index, edge_weight = remove_self_loops(edge_index=edge_index, edge_attr=edge_weight)
        deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=num_nodes)  # + 1e-10

        h_j = edge_weight.view(-1, 1) * h[edge_index[1]]
        aggr_out = scatter_add(h_j, edge_index[0], dim=0, dim_size=num_nodes)
        out = (deg.view(-1, 1) * self.lin1(x) + aggr_out) + self.lin2(x)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=True,
                 normalize=True, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.edge_mlp = nn.Sequential(nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),nn.Dropout(p=0),act_fn,
                                      nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp = nn.Sequential(nn.Linear(hidden_nf + input_nf, hidden_nf),nn.Dropout(p=0), act_fn,
                                      nn.Linear(hidden_nf, output_nf))
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class EGNNLayer(MessagePassing):
    """E(n)-equivariant Message Passing Layer
    Is currently not compatible with the Pytorch Geometric HeteroConv class, because are returning here
    only the updated target nodes features.
    TODO: Change this to conform with general Pytorch Geometric interface.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_features: int,
        out_features: int,
        act: nn.Module,
        dropout: float = 0.5,
        node_aggr: Aggregation = SumAggregation,
        cord_aggr: Aggregation = MeanAggregation,
        residual: bool = True,
        update_coords: bool = True,
        norm_coords: bool = True,
        norm_coors_scale_init: float = 1e-2,
        norm_feats: bool = True,
        initialization_gain: float = 1,
        return_pos: bool = True,
        attention = False
    ):
        super().__init__(aggr=None)
        self.node_aggr = node_aggr()
        self.cord_aggr = cord_aggr()
        self.residual = residual
        self.update_coords = update_coords
        self.act = act
        self.initialization_gain = initialization_gain
        self.return_pos = return_pos
        self.attention = attention

        if (node_features != out_features) and residual:
            raise ValueError(
                "Residual connections are only compatible with the same input and output dimensions."
            )

        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + 1, hidden_features),
            nn.Dropout(dropout),
            act,
            nn.Linear(hidden_features, hidden_features),
            act
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_features, 1),
                nn.Sigmoid()
            )

        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            nn.Dropout(dropout),
            act,
            nn.Linear(hidden_features, out_features),
        )

        layer = nn.Linear(hidden_features, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.pos_net = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            act,
            layer,
        )

        self.node_norm = (
            torch_geometric.nn.norm.LayerNorm(node_features) if norm_feats else nn.Identity()
        )
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coords else nn.Identity()
        )

        # self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            if (type(self.act) is nn.SELU):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear", mode="fan_in")
                nn.init.zeros_(module.bias)
            else:
                # seems to be needed to keep the network from exploding to NaN with greater depths
                nn.init.xavier_normal_(module.weight, gain=self.initialization_gain)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptPairTensor = None,
    ):
        # TODO: Think about a better solution for the residual connection
        if self.residual:
            residual = x if isinstance(x, Tensor) else x[1]
        x_dest, pos = self.propagate(edge_index, x=x, pos=pos, edge_attr=None, edge_weight=None)

        if self.residual:
            x_dest = x_dest + residual

        out = (x_dest, pos) if self.return_pos else x_dest
        return out

    def message(
        self, x_i: Tensor, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, edge_weight: OptTensor = None
    ):
        """Create messages"""
        pos_dir = pos_i - pos_j
        dist = torch.norm(pos_dir, dim=-1, keepdim=True)
        input = [self.node_norm(x_i), self.node_norm(x_j), dist]
        input = torch.cat(input, dim=-1)

        node_message = self.message_net(input)
        if self.attention:
            node_message = self.att_mlp(node_message) * node_message
        pos_message = self.coors_norm(pos_dir) * self.pos_net(node_message)
        if edge_weight is not None:
            node_message = node_message * edge_weight.unsqueeze(-1)
            pos_message = pos_message * edge_weight.unsqueeze(-1)

        return node_message, pos_message

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Tensor = None,
        dim_size: int = None,
    ) -> Tensor:
        node_message, pos_message = inputs
        agg_node_message = self.node_aggr(node_message, index, ptr, dim_size)
        agg_pos_message = self.cord_aggr(pos_message, index, ptr, dim_size)
        return agg_node_message, agg_pos_message

    def update(
        self,
        message: Tuple[Tensor, Tensor],
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
    ):
        node_message, pos_message = message
        x_, pos_ = (x, pos) if isinstance(x, Tensor) else (x[1], pos[1])
        input = torch.cat((x_, node_message), dim=-1)
        x_new = self.update_net(input)
        pos_new = pos_ + pos_message if self.update_coords else pos
        return x_new, pos_new

class CoorsNorm(nn.Module):
    """https://github.com/lucidrains/egnn-pytorch"""

    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale