import abc
from typing import List, Type

import torch
import torch_geometric
from torch import nn
from torch_geometric.nn import SumAggregation, Aggregation


def mlp_from_sizes(sizes: List[int], activation=nn.ReLU) -> nn.Sequential:
    if not sizes:
        raise ValueError("At least input dimension of MLP has to be given!")
    if len(sizes) == 1:
        return nn.Sequential()
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)

class PoolingLayer(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        """
        :param x: [num_nodes_total, embedding_dim]
        :param (batch): [num_nodes_total] (assignment of nodes to their batch element)
        :return: [batch_size, num_classes]
        """
        pass

class StandardPoolingLayer(PoolingLayer):
    def __init__(self, input_dim: int, num_classes: int, output_sizes: List[int], aggr: str):
        super().__init__()
        self.aggr: Aggregation = getattr(torch_geometric.nn, f"{aggr}Aggregation")()
        output_sizes = [input_dim] + output_sizes
        if output_sizes[-1] != num_classes:
            raise ValueError(f"Final network has output size {output_sizes[-1]} which is not equal to the number of "
                             f"classes {num_classes}!")
        self.out = mlp_from_sizes(output_sizes)


    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # [batch_size, num_classes]
        return self.out(self.aggr(x=x, index=batch))
class GraphSENNPool(PoolingLayer):
    def __init__(self, input_dim: int, num_classes: int, theta_sizes: List[int], h_sizes: List[int], aggr: str,
                 per_class_theta: bool, per_class_h: bool, global_theta: bool):
        super().__init__()
        if not per_class_h and not per_class_theta:
            raise ValueError("At least one of theta and h has to be per class. Otherwise, the same predictions would "
                             "need to be made for all classes.")

        theta_sizes = [input_dim * (2 if global_theta else 1)] + theta_sizes
        expected_theta_out = num_classes if per_class_theta else 1
        if theta_sizes[-1] != expected_theta_out:
            raise ValueError(f"Theta network has output size {theta_sizes[-1]} but expected {expected_theta_out}!")
        self.theta = mlp_from_sizes(theta_sizes)
        self.global_theta = global_theta

        h_sizes = [input_dim] + h_sizes
        expected_h_out = num_classes if per_class_h else 1
        if h_sizes[-1] != expected_h_out:
            raise ValueError(f"h network has output size {h_sizes[-1]} but expected {expected_h_out}!")
        self.h = mlp_from_sizes([input_dim] + h_sizes)

        self.g: Aggregation = getattr(torch_geometric.nn, f"{aggr}Aggregation")()


    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # [num_nodes_total, 1 | num_classes]
        h = self.h(x)
        if self.global_theta:
            # [batch_size, embedding_dim] pooled embeddings for each graph
            pooled = self.g(x, index=batch)
            # [num_nodes_total, embedding_dim] pooled embeddings for the graph of each nodes
            pooled = pooled[batch, :]
            # [num_nodes_total, 1 | num_classes]
            theta = self.theta(torch.cat((x, pooled), dim=-1))
        else:
            # [num_nodes_total, 1 | num_classes]
            theta = self.theta(x)
        # [batch_size, num_classes]
        x = self.g(x=h * theta, index=batch)
        return x