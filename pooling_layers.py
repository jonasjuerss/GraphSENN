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
from torch_geometric.utils import to_dense_adj

import custom_logger

if TYPE_CHECKING:
    from graph_senn import GraphSENN
    from decoders import GraphDecoder


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
    def forward(self, x: torch.Tensor, batch: torch.Tensor, annotations: Optional[torch.Tensor]) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        :param x: [num_nodes_total, embedding_dim]
        :param (batch): [num_nodes_total] (assignment of nodes to their batch element)
        :param annotations: [num_nodes_total, input_dim]
        :return:
            x: [batch_size, num_classes]
            theta (only for SENN Pooling)
            h (only for SENN Pooling)
        """
        pass

    @abc.abstractmethod
    def calculate_additional_losses(self, model: GraphSENN, x: torch.Tensor, x_out: torch.Tensor, batch: torch.Tensor,
                                    edge_index: torch.Tensor, theta: torch.Tensor, h: torch.Tensor,
                                    batch_size: int) ->\
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

    def calculate_additional_losses(self, model: GraphSENN, x: torch.Tensor, x_out: torch.Tensor, batch: torch.Tensor,
                                    edge_index: torch.Tensor, theta: torch.Tensor, h: torch.Tensor,
                                    batch_size: int) -> \
            Tuple[Union[torch.Tensor, int], dict]:
        return 0, {}
    def forward(self, x: torch.Tensor, batch: torch.Tensor, annotations: Optional[torch.Tensor]) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # [batch_size, num_classes]
        return self.out(self.aggr(x=x, index=batch)), None, None


class GraphSENNPool(PoolingLayer):
    def __init__(self, input_dim: int, num_classes: int, theta_sizes: List[int], h_sizes: List[int], aggr: str,
                 per_class_theta: bool, per_class_h: bool, global_theta: bool, theta_loss_weight: float,
                 feat_reconst_loss_weight: float, adj_reconst_loss_weight: float, decoder: Optional[GraphDecoder],
                 learn_h: bool):
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
        self.h = None
        if learn_h:
            self.h = mlp_from_sizes([input_dim] + h_sizes)
        elif theta_loss_weight != 0:
            raise ValueError("Cannot calculate theta loss when using ground-truth h as the derivative of h is unknown.")
        else:
            raise NotImplementedError("Ground truths for motifs not implemented yet!")
        self.per_class_h = per_class_h
        self.feat_loss_weight = feat_reconst_loss_weight
        self.adj_loss_weight = adj_reconst_loss_weight
        self.decoder = decoder
        self.decode_features_loss = torch.nn.MSELoss()
        self.decode_adj_loss = torch.nn.BCEWithLogitsLoss()  # BCELoss()

        self.g: Aggregation = getattr(torch_geometric.nn, f"{aggr}Aggregation")()
        self.num_classes = num_classes
        self.input_dim = input_dim

    def calculate_reconstruction_loss(self, x: torch.Tensor, batch: torch.Tensor, edge_index: torch.Tensor,
                                      h: torch.Tensor) ->\
            Tuple[torch.Tensor, torch.Tensor, float, float]:
        """

        :param x: [num_nodes_total, num_features_in]
        :param batch: [num_nodes_total]
        :param edge_index: [2, num_edges_total]
        :param h: [num_nodes_total, 1]
        :return:
        """
        # x_pred: [batch_size, max_num_nodes, num_features_in]
        # adj_pred [batch_size, max_num_nodes, max_num_nodes]
        # mask: [batch_size, max_num_nodes]
        x_pred, adj_pred, mask = self.decoder(h, edge_index, batch)
        feature_loss = self.decode_features_loss(x_pred[mask], x)

        # [batch_size, max_num_nodes, max_num_nodes]: boolean mask
        edge_mask = torch.logical_and(mask[:, None, :], mask[:, :, None])
        adj = to_dense_adj(edge_index, batch)
        masked_adj = adj[edge_mask]
        masked_adj_pred = adj_pred[edge_mask]
        adj_loss = self.decode_adj_loss(masked_adj_pred, masked_adj)
        # masked_adj should only have values of 0 or 1, we use > 0.5 only to convert it to boolean and avoid numerical
        # instability
        booleanized_adj = masked_adj > 0.5
        # Note that we do not use an activation but a loss from logits (necessary for numerical stability).
        # For accuracy we therefore want values > 0 as sigmoid(x) > 0.5 <=> x > 0
        edge_acc = torch.sum(booleanized_adj == (masked_adj_pred > 0)) / masked_adj.numel()
        # The sparsity gives us the edge accuracy if we just guess there are none
        sparsity = 1 - (torch.sum(booleanized_adj) / masked_adj.numel())
        return feature_loss, adj_loss, edge_acc.item(), sparsity.item()


    def calculate_stability_loss(self, model: GraphSENN, x_out: torch.Tensor, batch: torch.Tensor, edge_index: torch.Tensor,
                                 theta: torch.Tensor, batch_size: int):
        accum_loss = 0
        start_index = 0
        for sample in range(batch_size):
            # [num_nodes, num_features]
            mask = batch == sample
            # [num_nodes] containing the indices of all current nodes
            node_nums = torch.nonzero(mask).squeeze(1)
            x_out_sample = x_out[mask]
            num_nodes = x_out_sample.shape[0]
            input_dim = x_out_sample.shape[1]
            edge_index_sample = edge_index[:, torch.logical_and(edge_index[0] >= node_nums[0],
                                                                edge_index[0] <= node_nums[-1])] - start_index
            # [num_nodes, num_classes | 1]
            theta_sample = theta[mask]
            batch_sample = torch.zeros(num_nodes, dtype=torch.long, device=custom_logger.device)
            f = lambda emb: self(emb, batch_sample)[0]
            # [num_nodes, num_classes | 1, num_nodes * input_dim]
            J_x_h = jacobian(self.h, x_out_sample, create_graph=True, vectorize=True) \
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
            nabla_x_f_x = jacobian(f, x_out_sample, create_graph=True, vectorize=True).reshape(-1, num_nodes * input_dim)
            # Note that we take the divide the norm by the number of elements such that each sample has the same impact
            # which would otherwise not be given in our setting where the number of nodes/concepts can vary. It also
            # makes the choice of lambda more robust to the dataset.
            accum_loss = accum_loss + torch.norm(nabla_x_f_x - desired_derivative.squeeze(1)) / nabla_x_f_x.numel()
            start_index += num_nodes
        return accum_loss

    def calculate_additional_losses(self, model: GraphSENN, x: torch.Tensor, x_out: torch.Tensor, batch: torch.Tensor,
                                    edge_index: torch.Tensor, theta: torch.Tensor, h: torch.Tensor,
                                    batch_size: int) -> \
            Tuple[Union[torch.Tensor, int], dict]:
        res_dict = {}
        loss = 0
        if self.feat_loss_weight != 0 or self.adj_loss_weight != 0:
            feat_loss, adj_loss, edge_acc, sparsity = self.calculate_reconstruction_loss(x, batch, edge_index, h)
            loss = loss + self.feat_loss_weight * feat_loss + self.adj_loss_weight * adj_loss
            res_dict["feat_loss"] = feat_loss.item()
            res_dict["adj_loss"] = adj_loss.item()
            res_dict["edge_acc"] = edge_acc
            res_dict["sparsity"] = sparsity
        if self.theta_loss_weight != 0:
            stability_loss = self.calculate_stability_loss(model, x_out, batch, edge_index, theta, batch_size)
            loss = loss + self.theta_loss_weight * stability_loss
            res_dict["stability_loss"] = stability_loss.item()
        return loss, res_dict

    def forward(self, x: torch.Tensor, batch: torch.Tensor, annotations: torch.Tensor) -> \
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
