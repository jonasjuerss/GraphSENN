import abc
import warnings
from enum import Enum
from typing import List, Callable, Tuple

import torch.nn
import torch_geometric
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import to_dense_batch


class GraphDecoder(torch.nn.Module, abc.ABC):
    def __init__(self, gnn_sizes: List[int], input_data_dim: int, layer_type_name: str, gnn_activation):
        super().__init__()
        self.input_data_dim = input_data_dim

    @abc.abstractmethod
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param h: [num_nodes_total, num_classes | 1] the values of h for each class
        :param edge_index: [2, num_edges_total]
        :param batch: [num_nodes_total]
        :return:
            x: [batch_size, max_num_nodes
            adj_new: [batch_size, max_num_nodes, max_num_nodes]
            mask: [batch_size, max_num_nodes]
        """
        pass

class AdjGenerationType(Enum):
    IDENTITY = "identity"
    MLP = "mlp"
    GAE = "gae"
    MLP_GAE = "mlp-gae"

class EdgeConcatAdjNet(torch.nn.Module):
    def __init__(self, num_features_in: int, hidden_sizes: List[int], activation: torch.nn.Module):
        super().__init__()
        self.num_features_in = num_features_in
        mlp_sizes = [num_features_in * 2] + hidden_sizes + [1]
        layers = []
        for i in range(len(mlp_sizes) - 2):
            layers.append(torch.nn.Linear(mlp_sizes[i], mlp_sizes[i+1]))
            layers.append(activation)
        layers.append(torch.nn.Linear(mlp_sizes[-2], mlp_sizes[-1]))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        :param x: [batch_size, max_num_nodes, num_features_in]
        :param adj: [batch_size, max_num_nodes, max_num_nodes]
        :param mask: [batch_size, max_num_nodes]
        :return: new adjacency matrix: [batch_size, max_num_nodes, max_num_nodes]
        """
        # TODO use mask to avoid unnecessary layer calls
        # [batch_size, num_nodes_max, num_nodes_max, num_features_in]
        conc_shape = x.shape[:2] + x.shape[1:2] + x.shape[2:]
        # [batch_size, num_nodes_max, num_nodes_max, num_features_in] (manually broadcasting using expand)
        edge_pairwise = torch.cat((x[:, None, :, :].expand(conc_shape), x[:, :, None, :].expand(conc_shape)), dim=-1)
        # [batch_size, num_nodes_max, num_nodes_max, 1]
        adj_new = self.mlp(edge_pairwise)
        # print(f"{edge_pairwise.shape, adj_new.shape}, {torch.sigmoid(adj_new.min()).item():.2f}, {torch.sigmoid(adj_new.max()).item():.2f}")
        return adj_new.squeeze(-1)

class GAEAdjNet(torch.nn.Module):
    def __init__(self, num_features_in: int, hidden_sizes: List[int], activation: torch.nn.Module):
        super().__init__()
        self.num_features_in = num_features_in
        if hidden_sizes:
            mlp_sizes = [num_features_in] + hidden_sizes
            layers = []
            for i in range(len(mlp_sizes) - 2):
                layers.append(torch.nn.Linear(mlp_sizes[i], mlp_sizes[i+1]))
                layers.append(activation)
            layers.append(torch.nn.Linear(mlp_sizes[-2], mlp_sizes[-1]))
            self.mlp = torch.nn.Sequential(*layers)
        else:
            self.mlp = torch.nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, max_num_nodes, num_features_in]
        :param adj: [batch_size, max_num_nodes, max_num_nodes]
        :param mask: [batch_size, max_num_nodes]
        :return: new adjacency matrix: [batch_size, max_num_nodes, max_num_nodes]
        """
        # TODO use mask to avoid unnecessary layer calls
        # [batch_size, max_num_nodes, hidden_sizes[-1]]
        x = self.mlp(x)
        # [batch_size, max_num_nodes, max_num_nodes]
        adj = torch.bmm(x, x.transpose(1, 2))
        return adj

class FullyConnectedMessagePassingDecoder(GraphDecoder):
    def __init__(self, gnn_sizes: List[int], input_data_dim: int, layer_type_name: str, gnn_activation,
                 intermediate_adj: str, final_adj: str):
        super().__init__(gnn_sizes, input_data_dim, layer_type_name, gnn_activation)
        intermediate_adj = AdjGenerationType(intermediate_adj)
        final_adj = AdjGenerationType(final_adj)
        layer_type = getattr(torch_geometric.nn.dense, f"Dense{layer_type_name}")
        gnn_sizes = [1] + gnn_sizes[::-1] + [input_data_dim]
        gnn_layers = [layer_type(in_channels=gnn_sizes[i], out_channels=gnn_sizes[i+1])
                      for i in range(len(gnn_sizes) - 1)]
        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.activation = gnn_activation()
        def get_adj_layer(input_size: int, adj_type: AdjGenerationType) -> torch.nn.Module:
            if adj_type == AdjGenerationType.IDENTITY:
                return torch_geometric.nn.Sequential("x, adj, mask", [(torch.nn.Identity(), "adj -> adj")])
            elif adj_type == AdjGenerationType.MLP:
                return EdgeConcatAdjNet(input_size, [64], self.activation)
            elif adj_type == AdjGenerationType.GAE:
                return GAEAdjNet(input_size, [], self.activation)
            elif adj_type == AdjGenerationType.MLP_GAE:
                return GAEAdjNet(input_size, [64, 64], self.activation)
            else:
                raise NotImplementedError(f"There is currently no implementation for generating the adjacency matrix in"
                                          f" the {self.__class__.__name__} as {adj_type}!")
        self.adj_modules = torch.nn.ModuleList([get_adj_layer(s, intermediate_adj) for s in gnn_sizes[1:-2]] +
                                               [get_adj_layer(gnn_sizes[-2], final_adj)])
        if len(gnn_sizes) == 2:
            warnings.warn("No GNN layers were used. The network therefore cannot have any knowledge about adjacency and"
                          " adjacency reconstruction will be ignored!")
            self.adj_modules = []
            assert len(self.gnn_layers) == 0
        else:
            assert len(self.gnn_layers[:-1]) == len(self.adj_modules)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # On a fully connected graph, dense layers are way more efficient
        # x: [batch_size, num_nodes_max, 1]
        # mask: [batch_size, num_nodes_max]
        x, mask = to_dense_batch(h, batch)
        adj = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=h.device)
        # Only add edges between non-masked nodes (i.e. where the mask is True for both nodes)
        adj[torch.logical_and(mask[:, None, :], mask[:, :, None])] = 1
        for gnn_layer, adj_module in zip(self.gnn_layers[:-1], self.adj_modules):
            x = gnn_layer(x, adj, mask)
            x = self.activation(x)
            adj = adj_module(x, adj, mask)
        x = self.gnn_layers[-1](x, adj, mask)
        return x, adj, mask

