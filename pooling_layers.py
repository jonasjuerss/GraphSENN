from __future__ import annotations

import abc
from functools import partial
from typing import List, Tuple, Union, Optional
from typing import TYPE_CHECKING

import torch
import torch_geometric
from torch import nn
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch_geometric.nn import Aggregation

import custom_logger

if TYPE_CHECKING:
    from graph_senn import GraphSENN


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
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        :param x: [num_nodes_total, embedding_dim]
        :param (batch): [num_nodes_total] (assignment of nodes to their batch element)
        :return:
            x: [batch_size, num_classes]
            reg_loss: additional regularization loss
        """
        pass

    @abc.abstractmethod
    def calculate_additional_losses(self, model: GraphSENN, x: torch.Tensor, batch: torch.Tensor,
                                    edge_index: torch.Tensor, theta: torch.Tensor, batch_size: int) ->\
            Tuple[Union[torch.Tensor, int], dict]:
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

    def calculate_additional_losses(self, model: GraphSENN, x: torch.Tensor, batch: torch.Tensor,
                                    edge_index: torch.Tensor, theta: torch.Tensor, batch_size: int) -> \
            Tuple[Union[torch.Tensor, int], dict]:
        return 0, {}
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # [batch_size, num_classes]
        return self.out(self.aggr(x=x, index=batch)), None, None


class GraphSENNPool(PoolingLayer):
    def __init__(self, input_dim: int, num_classes: int, theta_sizes: List[int], h_sizes: List[int], aggr: str,
                 per_class_theta: bool, per_class_h: bool, global_theta: bool, theta_loss_weight: float):
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
        self.per_class_theta = per_class_theta
        self.theta_loss_weight = theta_loss_weight

        h_sizes = [input_dim] + h_sizes
        expected_h_out = num_classes if per_class_h else 1
        if h_sizes[-1] != expected_h_out:
            raise ValueError(f"h network has output size {h_sizes[-1]} but expected {expected_h_out}!")
        self.h = mlp_from_sizes([input_dim] + h_sizes)
        self.per_class_h = per_class_h

        self.g: Aggregation = getattr(torch_geometric.nn, f"{aggr}Aggregation")()
        self.num_classes = num_classes

    def calculate_stability_loss(self, model: GraphSENN, x: torch.Tensor, batch: torch.Tensor, edge_index: torch.Tensor,
                                 theta: torch.Tensor, batch_size: int):
        accum_loss = 0
        start_index = 0
        for sample in range(batch_size):
            # [num_nodes, num_features]
            mask = batch == sample
            # [num_nodes] containing the indices of all current nodes
            node_nums = torch.nonzero(mask).squeeze(1)
            x_sample = x[mask]
            num_nodes = x_sample.shape[0]
            input_dim = x_sample.shape[1]
            edge_index_sample = edge_index[:, torch.logical_and(edge_index[0] >= node_nums[0],
                                                                edge_index[0] <= node_nums[-1])] - start_index
            # [num_nodes, num_classes | 1]
            theta_sample = theta[mask]
            batch_sample = torch.zeros(num_nodes, dtype=torch.long, device=custom_logger.device)
            h = lambda emb: self.h(model.gnn_part(emb, edge_index=edge_index_sample, batch=batch_sample))
            f = lambda emb: model(emb, edge_index_sample, batch_sample)[0]
            # [num_nodes, num_classes | 1, num_nodes * input_dim]
            J_x_h = jacobian(h, x_sample, create_graph=True, vectorize=True) \
                .reshape(num_nodes, -1, num_nodes * input_dim)

            # [num_classes | 1, 1, num_nodes] @ [num_classes | 1, num_nodes, num_nodes * input_dim]
            # where the first dimension is interpreted as batch dimension, so we calculate
            # [1, num_nodes] @ [num_nodes, num_nodes * input_dim] for each of them. Note that either theta or h must
            # have a per-class output (otherwise predictions for all classes would be equal). Therefore, the broadcasted
            # first dimension must be num_classes. For each class, we want to regularize the derivative
            # $\theta(x)^TJ_x^h(x)$ to be close to $\nabla_xf(x)$
            # desired_derivative: [num_classes, 1, num_nodes * input_dim]
            desired_derivative = torch.matmul(theta_sample.T[:, None, :], J_x_h.transpose(0, 1))

            # [num_classes, num_nodes * input_dim]
            nabla_x_f_x = jacobian(f, x_sample, create_graph=True, vectorize=True).reshape(-1, num_nodes * input_dim)
            # Note that we take the divide the norm by the number of elements such that each sample has the same impact
            # which would otherwise not be given in our setting where the number of nodes/concepts can vary. It also
            # makes the choice of lambda more robust to the dataset.
            accum_loss = accum_loss + torch.norm(nabla_x_f_x - desired_derivative.squeeze(1)) / nabla_x_f_x.numel()
            start_index += num_nodes
        return accum_loss

    def calculate_additional_losses(self, model: GraphSENN, x: torch.Tensor, batch: torch.Tensor,
                                    edge_index: torch.Tensor, theta: torch.Tensor, batch_size: int) -> \
            Tuple[Union[torch.Tensor, int], dict]:
        res_dict = {}
        loss = 0
        if self.theta_loss_weight != 0:
            stability_loss = self.calculate_stability_loss(model, x, batch, edge_index, theta, batch_size)
            loss += self.theta_loss_weight * stability_loss
            res_dict["stability_loss"] = stability_loss.item()
        return loss, res_dict

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # [num_nodes_total, 1 | num_classes]
        h = self.h(x)
        theta_in = x
        if self.global_theta:
            # [batch_size, embedding_dim] pooled embeddings for each graph
            pooled = self.g(x, index=batch)
            # [num_nodes_total, embedding_dim] pooled embeddings for the graph of each node
            pooled = pooled[batch, :]
            theta_in = torch.cat((x, pooled), dim=-1)
        # [num_nodes_total, 1 | num_classes]
        theta = self.theta(theta_in)
        # [batch_size, num_classes]
        out = self.g(x=h * theta, index=batch)
        return out, theta, h
