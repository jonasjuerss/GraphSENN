"""
The custom motif dataset is copied from my MPhil Thesis
"""
from __future__ import annotations

import json
from typing import Union, Tuple, List, Optional

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from abc import ABC, abstractmethod

import motif_dataset.serializer as cs
from motif_dataset.graphutils import adj_to_edge_index
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric

class SparseGraph:
    def __init__(self, x: Tensor, edge_index: Tensor, annotations: Optional[Tensor] = None):
        self.x = x
        self.edge_index = edge_index
        self.annotations = annotations

    def merged_with(self, other: SparseGraph, merge_nodes: Union[Tuple[int, int], None] = None) -> SparseGraph:
        annotations = None
        if merge_nodes is None:
            x = torch.cat((self.x, other.x), dim=0)
            if self.annotations is not None:
                annotations = torch.cat((self.annotations, other.annotations), dim=0)
            edge_index = torch.cat((self.edge_index, other.edge_index + self.num_nodes()), dim=1)
        else:
            # leave out the merged node from new
            x = torch.cat((self.x, other.x[:merge_nodes[1], :], other.x[merge_nodes[1] + 1:, :]), dim=0)
            if self.annotations is not None:
                annotations = torch.cat((self.annotations, other.annotations[:merge_nodes[1], :],
                                         other.annotations[merge_nodes[1] + 1:, :]), dim=0)
            # adjust edge_index of new graph accordingly. Will temporarily yield negative node indices because the node will be in x_base
            edge_index = other.edge_index.copy()
            edge_index[edge_index == merge_nodes[1]] = merge_nodes[0] - self.num_nodes()
            edge_index[edge_index > merge_nodes[1]] -= 1  # adjust all node indices after the removed node
            edge_index = torch.cat((self.edge_index, edge_index + self.num_nodes()), dim=1)
        return SparseGraph(x, edge_index, annotations)

    def add_nodes(self, features: Tensor, annotations: Optional[Tensor] = None) -> None:
        if len(features.shape) == 1:
            features = features[None, :]
        elif len(features.shape) != 2:
            raise ValueError(f"Expected feature vector to be of dimension 1 or 2 but got {len(features.shape)}!")

        if features.shape[1] == self.x.shape[1]:
            self.x = torch.cat((self.x, features), dim=1)
            if self.annotations is not None:
                self.annotations = torch.cat((self.annotations, annotations), dim=1)
        else:
            raise ValueError(f"Expected {self.x.shape[1]} features but got {features.shape[0]}!")

    def add_edges(self, edges: List[List[int]]) -> None:
        """
        :param edges: Note: this is a list of edges (e.g. [[0, 1], [1, 0]]) would add a bidirectional edge between nodes
        0 and 1. It is not a slice of an edge_index tensor
        """
        self.edge_index = torch.cat((self.edge_index, torch.tensor(edges, dtype=torch.long).T), dim=1)

    def add_edges_if_not_exist_list(self, edges: List[List[int]]) -> None:
        """
        :param edges: Note: this is a list of edges (e.g. [[0, 1], [1, 0]]) would add a bidirectional edge between nodes
        0 and 1. It is not a slice of an edge_index tensor
        """
        # [2, num_edges]
        added_edge_index = torch.tensor(edges, dtype=torch.long).T
        self.add_edges_if_not_exist_edge_index(added_edge_index)

    def add_edges_if_not_exist_edge_index(self, edge_index: Tensor) -> None:
        for i in range(edge_index.shape[1]):
            if not torch.any(torch.all(self.edge_index == edge_index[:, i:i+1], dim=0)):
                self.edge_index = torch.cat((self.edge_index, edge_index[:, i:i+1]), dim=1)

    def num_nodes(self):
        return self.x.shape[0]

    def render(self):
        data = Data(x=self.x, edge_index=self.edge_index)
        g = torch_geometric.utils.to_networkx(data, to_undirected=True)
        nx.draw(g, with_labels=True)
        plt.show()


class Motif(cs.ArgSerializable):
    def __init__(self, num_colors: int, max_nodes: int, args: dict):
        """

        :param num_colors: The total number of colors in the graph
        :param max_nodes: the maximum number of nodes a sample of this motif can have
        :param args: All keyword arguments the constructor of the subclass took. This is used for serialization
        """
        super().__init__(args)
        self.num_colors = num_colors
        self.max_nodes = max_nodes

    @abstractmethod
    def sample(self) -> SparseGraph:
        pass

    @property
    def name(self):
        return self.__class__.__name__[:-5]


class HouseMotif(Motif):

    def __init__(self, roof_colors: List[int], basement_colors: List[int], num_colors: int, roof_annotation: int = 0,
                 basement_annotation: int = 0):
        """
          0
         / \
        1---2
        |   |
        3---4
        :param roof_colors: Possible colors for the three roof nodes (will be chosen uniformly at random on sample())
        :param basement_colors: Possible colors for the two basement nodes (will be chosen uniformly at random on sample())
        :param num_colors: number of different colors in the overall graph
        """
        super().__init__(num_colors, 5,
                         dict(roof_colors=roof_colors, basement_colors=basement_colors, num_colors=num_colors,
                              roof_annotation=roof_annotation, basement_annotation=basement_annotation))
        self.roof_colors = roof_colors
        self.basement_colors = basement_colors
        self.roof_annotation = roof_annotation
        self.basement_annotation = basement_annotation

    def sample(self) -> SparseGraph:
        roof_color = _random_list_entry(self.roof_colors)
        basement_color = _random_list_entry(self.basement_colors)
        x = torch.zeros((5, self.num_colors))
        x[:3, roof_color] = 1
        x[3:, basement_color] = 1
        annotations = torch.empty((5,), dtype=torch.long)
        annotations[:3] = self.roof_annotation
        annotations[3:] = self.basement_annotation
        edge_index = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]], dtype=torch.long).T
        return SparseGraph(x, to_undirected(edge_index), annotations)


# class TriangleMotif(Motif):
# CONTAINED IN FULLY CONNECTED
#     def __init__(self, colors: List[int], num_colors: int):
#         super().__init__(num_colors, 3, dict(colors=colors, num_colors=num_colors))
#         self.colors = colors
#
#     def sample(self) -> SparseGraph:
#         #   0
#         #  / \
#         # 1---2
#         color = _random_list_entry(self.colors)
#         x = torch.zeros((3, self.num_colors))
#         x[:, color] = 1
#         edge_index = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long).T
#         return SparseGraph(x, to_undirected(edge_index))


class FullyConnectedMotif(Motif):
    def __init__(self, num_nodes: int, colors: List[int], num_colors: int, annotation: int = 0):
        super().__init__(num_colors, num_nodes, dict(num_nodes=num_nodes, colors=colors, num_colors=num_colors,
                                                     annotation=annotation))
        self.colors = colors
        self.num_nodes = num_nodes
        self.annotation = annotation

    def sample(self) -> SparseGraph:
        color = _random_list_entry(self.colors)
        x = torch.zeros((self.num_nodes, self.num_colors))
        x[:, color] = 1
        adj = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.long)
        node_indices = torch.arange(self.num_nodes)
        adj[node_indices, node_indices] = 0
        edge_index, _, _ = adj_to_edge_index(adj)
        return SparseGraph(x, edge_index, torch.ones(x.shape[0], dtype=torch.long) * self.annotation)

    @property
    def name(self):
        if self.num_nodes == 3:
            return "Triangle"
        elif self.num_nodes == 4:
            return "FCSquare"
        elif self.num_nodes == 5:
            return "FCPentagon"
        elif self.num_nodes == 6:
            return "FCHexagon"


class BinaryTreeMotif(Motif):
    def __init__(self, max_depth: int, colors: List[int], num_colors: int, random: bool = True, annotation: int = 0):
        super().__init__(num_colors, (2 ** (max_depth + 1)) - 1,
                         dict(max_depth=max_depth, colors=colors, num_colors=num_colors, random=random))
        self.max_depth = max_depth
        self.colors = colors
        self.random = random
        self.annotation = annotation

    def sample(self):
        """
        Idea: each path from root to leaves has length 1,...,max_depth with same probability
        """
        color = _random_list_entry(self.colors)
        tree = self._random_binary_tree(self.max_depth, color)
        tree.annotations = torch.ones(tree.num_nodes(), dtype=torch.long) * self.annotation
        return tree

    def _random_binary_tree(self, max_depth: int, color: int) -> SparseGraph:
        x = torch.zeros((1, self.num_colors))
        x[:, color] = 1
        edge_index = torch.tensor([[], []], dtype=torch.long)
        result = SparseGraph(x, edge_index)
        if self.max_depth == 0:
            return result
        if not self.random or torch.rand(()) < (max_depth - 1) / max_depth:
            left = self._random_binary_tree(max_depth - 1, color)
            result = result.merged_with(left)
            result.add_edges([[0, 1], [1, 0]])
        if not self.random or torch.rand(()) < (max_depth - 1) / max_depth:
            num_nodes = result.num_nodes()
            right = self._random_binary_tree(max_depth - 1, color)
            result = result.merged_with(right)
            result.add_edges([[0, num_nodes], [num_nodes, 0]])
        return result


def _random_list_entry(list: List[int]):
    return list[torch.randint(len(list), (1,))]
