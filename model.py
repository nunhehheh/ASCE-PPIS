from ASAP import *
import math
import torch
import torch.nn as nn
import torch_geometric
from torch import Tensor
import numpy as np
from torch_geometric.nn import (
    Aggregation,
    MeanAggregation,
    MessagePassing,
    SumAggregation,
)
import numpy as np
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
SEED = 2024
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
MAP_CUTOFF = 14
DIST_NORM = 15
MAP_TYPE = "d"
INPUT_DIM = 62
p = 0.3 #pooling rate
HIDDEN_DIM = 128
LAYER = 8  # the number of EGCL layers
DROPOUT = 0.5
LEARNING_RATE = 5E-4
WEIGHT_DECAY = 1E-5
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 60
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def extract_edge(edges, idx): #extract subgraph by index
    edge_start = edges[0].tolist()
    edge_end = edges[1].tolist()
    idx = idx.tolist()
    idx_set = set(idx)
    filtered_edges = [(s, e) for s, e in zip(edge_start, edge_end) if s in idx_set and e in idx_set]
    new_idx_map = {old: new for new, old in enumerate(idx)}
    new_edge_start = torch.tensor([new_idx_map[s] for s, _ in filtered_edges], device=device).long()
    new_edge_end = torch.tensor([new_idx_map[e] for _, e in filtered_edges], device=device).long()
    edges = [new_edge_start, new_edge_end]
    edge_index = torch.stack(edges, dim=0)
    return edge_index

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


class EGNNLayer(MessagePassing):
    """E(n)-equivariant Message Passing Layer"""

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


class EGNNGlobalNodeHetero(nn.Module):
    """E(n)-equivariant Message Passing Network"""

    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        num_layers,
        act=nn.SiLU(),
        dropout=0.5,
        node_aggr=SumAggregation,
        cord_aggr=MeanAggregation,
        update_coords=True,
        residual=True,
        norm_coords=True,
        norm_coors_scale_init=1e-2,
        norm_feats=True,
        initialization_gain=1,
        weight_share=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.weight_share = weight_share
        if weight_share:
            # Use a single layer that will be shared across all iterations
            self.shared_layer = self.create_layer(
                node_features,
                edge_features,
                hidden_features,
                out_features,
                act,
                dropout,
                node_aggr,
                cord_aggr,
                update_coords,
                residual,
                norm_coords,
                norm_coors_scale_init,
                norm_feats,
                initialization_gain,
            )
        else:
            # Create a list of layers, one for each iteration
            self.layers = nn.ModuleList(
                [
                    self.create_layer(
                        node_features,
                        edge_features,
                        hidden_features,
                        out_features,
                        act,
                        dropout,
                        node_aggr,
                        cord_aggr,
                        update_coords,
                        residual,
                        norm_coords,
                        norm_coors_scale_init,
                        norm_feats,
                        initialization_gain,
                    )
                    for _ in range(num_layers)
                ]
            )

    def create_layer(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        act,
        dropout,
        node_aggr,
        cord_aggr,
        update_coords,
        residual,
        norm_coords,
        norm_coors_scale_init,
        norm_feats,
        initialization_gain,
    ):
        # Centralized layer creation logic
        return nn.ModuleDict(
            {
                "atom_to_atom": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=False,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=False,
                    initialization_gain=initialization_gain,
                    attention=True
                ),
                "atom_to_global_node": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                    #attention=True
                ),
                "global_node_to_global_node": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                    #attention=True
                ),
                "global_node_to_atom": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                    #attention=True
                ),
            }
        )


    def forward(
        self,
        x_atom,
        pos_atom,
        x_global_node,
        pos_global_node,
        edge_index_atom_atom,
        edge_index_atom_global_node,
        edge_index_global_node_global_node,
        edge_index_global_node_atom,
    ):

        for i in range(self.num_layers):
            layer = self.shared_layer if self.weight_share else self.layers[i]
            x_atom, pos_atom = layer["atom_to_atom"](
                x=(x_atom, x_atom),
                edge_index=edge_index_atom_atom,
                pos=(pos_atom, pos_atom),
            )
            x_global_node, pos_global_node = layer["atom_to_global_node"](
                x=(x_atom, x_global_node),
                edge_index=edge_index_atom_global_node,
                pos=(pos_atom, pos_global_node),
            )
            x_global_node, pos_global_node = layer["global_node_to_global_node"](
                x=(x_global_node, x_global_node),
                edge_index=edge_index_global_node_global_node,
                pos=(pos_global_node, pos_global_node),
            )
            x_atom, pos_atom = layer["global_node_to_atom"](
                x=(x_global_node, x_atom),
                edge_index=edge_index_global_node_atom,
                pos=(pos_global_node, pos_atom),
            )

        return x_atom, x_global_node, pos_atom, pos_global_node

class ASCEPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass):
        super(ASCEPPIS, self).__init__()
        self.embedding_in = nn.Linear(nfeat, nhidden)
        self.embedding_out = nn.Linear(nhidden, nclass)
        self.model = EGNNGlobalNodeHetero(node_features=nhidden, edge_features=0, hidden_features=nhidden,
                                          num_layers=nlayers,
                                          out_features=nhidden, act=nn.SiLU(),dropout=0.0, weight_share=False)
        self.pools = ASAP_Pooling(in_channels=nhidden)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=5,
                                                                    min_lr=1e-6)

    def forward(self, node_feat, node_pos, edge_index):
        node_feat = self.embedding_in(node_feat)
        scores = self.pools(node_feat,edge_index,node_pos,clean=True)
        _, idx = torch.topk(scores.T, int(p * node_feat.shape[0]))
        idx, _ = torch.sort(idx)
        scores = scores.unsqueeze(-1)
        x_atom = scores * node_feat
        h_pool = x_atom[idx]
        edges_pool = extract_edge(edge_index, idx)
        x_pool = node_pos[idx]
        virtual_node_feat = h_pool
        virtual_node_feat = virtual_node_feat.squeeze(1)
        virtual_node_pos = x_pool
        edges_to_remove = (edge_index[0] == idx.unsqueeze(1)).any(dim=0)
        edges_to_keep = (edge_index[1] == idx.unsqueeze(1)).any(dim=0)
        final_edges_to_keep = ~edges_to_remove & edges_to_keep
        A2V_edge_index = edge_index[:, final_edges_to_keep]
        A2V_edge_index[1] = discretized_flat(A2V_edge_index[1])
        V2A_edge_index = torch.flip(A2V_edge_index, [0])
        h, x_global_node, pos_atom, pos_global_node = self.model(node_feat, node_pos, virtual_node_feat,
                                                                 virtual_node_pos, edge_index, A2V_edge_index,edges_pool,
                                                                 V2A_edge_index)
        out = self.embedding_out(h)
        return out

def discretized_flat(idx):
    min_val = idx.min()
    max_val = idx.max()
    num_bins = len(torch.unique(idx))
    bins = torch.linspace(min_val, max_val,steps=num_bins).to(device)
    discretized = torch.bucketize(idx, bins)
    return discretized
