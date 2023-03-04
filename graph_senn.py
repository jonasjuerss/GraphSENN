from typing import List, Type

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from pooling_layers import PoolingLayer


class GraphSENN(torch.nn.Module):
    def __init__(self, gnn_sizes: List[int], input_dim: int, output_dim: int, layer_type: Type[torch.nn.Module],
                 gnn_activation, pooling_layer: PoolingLayer, **gnn_layer_kwargs):
        super().__init__()
        self.activation = gnn_activation
        layer_sizes = [input_dim] + gnn_sizes
        self.gcn_layers = torch.nn.ModuleList([
            layer_type(in_channels=layer_sizes[i], out_channels=layer_sizes[i+1], **gnn_layer_kwargs)
            for i in range(len(layer_sizes) - 1)])
        self.output_dim = output_dim
        self.pooling_layer = pooling_layer
        self.output_layer = torch.nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.gcn_layers[:-1]:
            x = self.activation(layer(x=x, edge_index=edge_index))
        #
        x = self.gcn_layers[-1](x=x, edge_index=edge_index)
        # [batch_size, layer_sizes[-1]]
        x = self.pooling_layer(x, batch)
        return F.log_softmax(x, dim=-1)
