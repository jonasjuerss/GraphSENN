from functools import partial
from typing import List, Type, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data

from pooling_layers import PoolingLayer

class GumbleSoftmaxLayer(torch.nn.Module):
    def __init__(self, hard: bool):
        super().__init__()
        self.hard = hard
    def forward(self, x):
        return torch.nn.functional.gumbel_softmax(x, hard=self.hard)
class GraphSENN(torch.nn.Module):
    def __init__(self, gnn_sizes: List[int], input_dim: int, output_dim: int, layer_type: Type[torch.nn.Module],
                 gnn_activation, pooling_layer: PoolingLayer, concept_activation: str, **gnn_layer_kwargs):
        super().__init__()
        self.activation = gnn_activation
        self.input_dim = input_dim
        layer_sizes = [input_dim] + gnn_sizes
        gnn_layers = []
        for i in range(len(layer_sizes) - 2):
            gnn_layers.append((layer_type(in_channels=layer_sizes[i], out_channels=layer_sizes[i+1], **gnn_layer_kwargs),
                               'x, edge_index -> x'))
            gnn_layers.append((self.activation(), 'x -> x'))
        if len(layer_sizes) > 1:
            gnn_layers.append((layer_type(in_channels=layer_sizes[-2], out_channels=layer_sizes[-1], **gnn_layer_kwargs),
                               'x, edge_index -> x'))
        else:
            gnn_layers = [(torch.nn.Identity(), 'x -> x')]



        if concept_activation == "sigmoid":
            gnn_layers.append((torch.nn.Sigmoid(), 'x -> x'))
        if concept_activation == "softmax":
            gnn_layers.append((torch.nn.Softmax(), 'x -> x'))
        elif concept_activation == "gumbel_softmax":
            gnn_layers.append((GumbleSoftmaxLayer(True), 'x -> x'))
        elif concept_activation == "gumbel_softmax_soft":
            gnn_layers.append((GumbleSoftmaxLayer(False), 'x -> x'))
        elif concept_activation != "none":
            raise ValueError(f"Unknown concept activation function {concept_activation}")

        self.gnn_part = torch_geometric.nn.Sequential('x, edge_index, batch', gnn_layers)
        self.output_dim = output_dim
        self.pooling_layer = pooling_layer
        self.output_layer = torch.nn.Linear(layer_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, annotations: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param data:
        :return:
            x: [batch_size, num_classes] log_softmax predictions of the classes
            reg_loss: additional regularization loss
        """
        x = self.gnn_part(x, edge_index, batch)
        # [batch_size, num_classes]
        x_out, theta, h = self.pooling_layer(x, batch, annotations)
        return F.log_softmax(x_out, dim=-1), x, theta, h
