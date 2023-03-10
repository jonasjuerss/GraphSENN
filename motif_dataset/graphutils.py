"""
The custom motif dataset is copied from my MPhil Thesis
"""
import math
from typing import Tuple, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj, to_dense_batch
from torch_scatter import scatter

import custom_logger


def adj_to_edge_index(adj: torch.Tensor, mask: Optional[torch.Tensor] = None)\
        -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """

    :param adj: [max_num_nodes, max_num_nodes] or [batch_size, max_num_nodes, max_num_nodes]
    :param mask: None or [max_num_nodes] or [batch_size, max_num_nodes]
    :return:
        edge_index: [2, num_edges]
        batch: in case of back dimension: [num_nodes_total]
        num_nodes: total number of nodes
    """
    if adj.ndim == 2:
        if mask is not None:
            # There should be an easier wys to index values at the mask
            adj = adj[torch.logical_and(mask[0][None, :], mask[0][:, None])]
            num_nodes = math.isqrt(adj.shape[0])
            adj = adj.view(num_nodes, num_nodes)
        return adj.nonzero().t().contiguous(), None, adj.shape[-1]
    elif adj.ndim == 3:
        masks = torch.logical_and(mask[:, None, :], mask[:, :, None])
        num_nodes = 0
        edge_index = torch.empty(2, 0, device=custom_logger.device)
        batch = torch.empty(0, device=custom_logger.device, dtype=torch.long)
        for i in range(adj.shape[0]):
            cur_adj = adj[i][masks[i, :, :]]
            cur_nodes = math.isqrt(cur_adj.shape[0])
            cur_adj = cur_adj.view(cur_nodes, cur_nodes)
            edge_index = torch.cat((edge_index, cur_adj.nonzero().t().contiguous() + num_nodes), dim=1)
            batch = torch.cat((batch, i * torch.ones(cur_nodes, device=custom_logger.device, dtype=torch.long)), dim=0)
            num_nodes += cur_nodes
        return edge_index, batch, num_nodes
    else:
        raise ValueError(f"Unsupported number of dimensions: {adj.ndim}. The only supported formats are "
                         f"[num_nodes, num_nodes] and [batch_size, max_num_nodes, max_num_nodes]!")

