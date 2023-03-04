from typing import List, Type

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GCNConv, SumAggregation
from torch_geometric.nn.models.basic_gnn import BasicGNN
import torch.nn.functional as F

from graph_senn_layer import GraphSENNLayer

class GraphSENN(torch.nn.Module):
    def __init__(self, layer_sizes: List[int], input_dim: int, output_dim: int, layer_type: Type[torch.nn.Module], gnn_activation,
                 **layer_kwargs):
        super().__init__()
        self.activation = gnn_activation
        layer_sizes = [input_dim] + layer_sizes
        self.gcn_layers = torch.nn.ModuleList([
            layer_type(in_channels=layer_sizes[i], out_channels=layer_sizes[i+1], **layer_kwargs)
            for i in range(len(layer_sizes) - 1)])
        self.output_dim = output_dim
        # self.output_layer = GraphSENNLayer(layer_sizes[-1], output_dim)
        self.aggr = SumAggregation()
        self.output_layer = torch.nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.gcn_layers[:-1]:
            x = self.activation(layer(x=x, edge_index=edge_index))
        #
        x = self.gcn_layers[-1](x=x, edge_index=edge_index)
        # [batch_size, layer_sizes[-1]]
        x = self.aggr(x=x, index=batch)
        x = self.output_layer(x)
        #x = self.senn_layer()
        return F.log_softmax(x, dim=-1)
