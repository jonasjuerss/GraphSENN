import abc

import numpy as np
import torch_geometric
from torch_geometric.datasets import TUDataset

from motif_dataset.motif_dataset import UniqueMotifCategorizationDataset
from motif_dataset.motifs import BinaryTreeMotif, HouseMotif, FullyConnectedMotif


class DatasetWrapper(abc.ABC):
    def __init__(self, dataset, num_classes: int, num_node_features: int):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_node_features = num_node_features

    def get_node_labels(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: [batch_size, input_dim] input features
        :return: [batch_size] string array of corresponding node labels
        """
        return np.zeros(x.shape[0], dtype=str)

    def get_node_colors(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: [batch_size, input_dim] input features
        :return: [batch_size] string array of corresponding node colors as hex strings
        """
        return np.repeat("#1f77b4", x.shape[0])


class MutagWrapper(DatasetWrapper):
    def __init__(self):
        dataset = TUDataset(root='/tmp', name='MUTAG')
        super().__init__(dataset, dataset.num_classes, dataset.num_node_features)
        self.label_map = np.array(['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca'])
        self.color_map = np.array(
            ['#2c3e50', '#e74c3c', '#27ae60', '#3498db', '#CDDC39', '#f39c12', '#795548', '#8e44ad', '#3F51B5',
             '#7f8c8d', '#e84393', '#607D8B', '#8e44ad', '#009688'])

    def get_node_labels(self, x: np.ndarray) -> np.ndarray:
        return self.label_map[np.argmax(x, axis=-1)]

    def get_node_colors(self, x: np.ndarray) -> np.ndarray:
        return self.color_map[np.argmax(x, axis=-1)]


class EnzymesWrapper(DatasetWrapper):
    def __init__(self):
        dataset = TUDataset(root='/tmp', name='ENZYMES')
        super().__init__(dataset, dataset.num_classes, dataset.num_node_features)


class RedditBinaryWrapper(DatasetWrapper):
    def __init__(self):
        dataset = TUDataset(root='/tmp', name='REDDIT-BINARY', transform=torch_geometric.transforms.Constant())
        super().__init__(dataset, dataset.num_classes, dataset.num_node_features)

class UniqueMotifWrapper(DatasetWrapper):
    def __init__(self, num_samples=2000, num_colors=2):
        sampler = UniqueMotifCategorizationDataset(BinaryTreeMotif(5, [0], num_colors),
                                                   [HouseMotif([1], [1], num_colors),
                                                    FullyConnectedMotif(5, [1], num_colors)],
                                                   [[0.4, 0.6], [0.4, 0.6]])
        super().__init__([sampler.sample() for _ in range(num_samples)], sampler.num_classes, sampler.num_node_features)
        self.color_map = np.array(['#2c3e50', '#e74c3c'])

    def get_node_colors(self, x: np.ndarray) -> np.ndarray:
        return self.color_map[np.argmax(x, axis=-1)]


class UniqueMotifEasyWrapper(DatasetWrapper):
    def __init__(self, num_samples=2000, num_colors=3):
        sampler = UniqueMotifCategorizationDataset(BinaryTreeMotif(5, [0], num_colors),
                                                   [HouseMotif([1], [1], num_colors),
                                                    FullyConnectedMotif(5, [2], num_colors)],
                                                   [[0.4, 0.6], [0.4, 0.6]])
        super().__init__([sampler.sample() for _ in range(num_samples)], sampler.num_classes, sampler.num_node_features)
        self.color_map = np.array(['#2c3e50', '#e74c3c'])

    def get_node_colors(self, x: np.ndarray) -> np.ndarray:
        return self.color_map[np.argmax(x, axis=-1)]


__all__ = [MutagWrapper, EnzymesWrapper, RedditBinaryWrapper, UniqueMotifWrapper]


def from_name(name: str):
    for w in __all__:
        if w.__name__[:-7].lower() == name.replace("-", "").lower():
            return w()
    raise ValueError(f"Could not find wrapper matching dataset name {name}")
