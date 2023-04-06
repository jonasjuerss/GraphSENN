import abc
from typing import Optional, List

import numpy as np
import torch_geometric
from torch_geometric.datasets import TUDataset

from motif_dataset.motif_dataset import UniqueMotifCategorizationDataset, \
    UniqueMultipleOccurrencesMotifCategorizationDataset
from motif_dataset.motifs import BinaryTreeMotif, HouseMotif, FullyConnectedMotif


class DatasetWrapper(abc.ABC):
    def __init__(self, dataset, num_classes: int, num_node_features: int, class_names: Optional[List[str]] = None):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            if len(class_names) != num_classes:
                raise ValueError(f"Got {len(class_names)} class names for {num_classes} classes!")
            self.class_names = class_names

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
        dataset = TUDataset(root='/tmp', name='MUTAG').shuffle()
        super().__init__(dataset, dataset.num_classes, dataset.num_node_features, ["not mutagenic", "mutagenic"])
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
        dataset = TUDataset(root='/tmp', name='ENZYMES').shuffle()
        super().__init__(dataset, dataset.num_classes, dataset.num_node_features)


class RedditBinaryWrapper(DatasetWrapper):
    def __init__(self):
        dataset = TUDataset(root='/tmp', name='REDDIT-BINARY', transform=torch_geometric.transforms.Constant()).shuffle()
        super().__init__(dataset, dataset.num_classes, dataset.num_node_features)

class MotifWrapper(DatasetWrapper, abc.ABC):
    
    def __init__(self, sampler, num_samples=2000):
        super().__init__([sampler.sample() for _ in range(num_samples)],
                         sampler.num_classes, sampler.num_node_features, sampler.class_names)
        self.color_map = np.array(
            ['#2c3e50', '#e74c3c', '#27ae60', '#3498db', '#CDDC39', '#f39c12', '#795548', '#8e44ad', '#3F51B5',
             '#7f8c8d', '#e84393', '#607D8B', '#8e44ad', '#009688'])

    def get_node_colors(self, x: np.ndarray) -> np.ndarray:
        """

        :param x: one-hot features [num_nodes, num_colors] OR color indices [num_nodes]
        :return:
        """
        if x.ndim == 2:
            x = np.argmax(x, axis=-1)
        return self.color_map[x]
    
class UniqueMotifWrapper(MotifWrapper):
    def __init__(self, num_colors=2):
        sampler = UniqueMultipleOccurrencesMotifCategorizationDataset(BinaryTreeMotif(5, [0], num_colors),
                                                                      [HouseMotif([1], [1], num_colors, 1, 1),
                                                                       FullyConnectedMotif(5, [1], num_colors, 2)],
                                                                      [[0.4, 0.6], [0.4, 0.6]])
        super().__init__(sampler)


class UniqueMotifHardColorWrapper(MotifWrapper):
    def __init__(self, num_colors=2):
        sampler = UniqueMultipleOccurrencesMotifCategorizationDataset(BinaryTreeMotif(5, [0], num_colors),
                                                                      [HouseMotif([1], [1], num_colors, 1, 1),
                                                                       FullyConnectedMotif(5, [1], num_colors, 2),
                                                                       FullyConnectedMotif(3, [1], num_colors, 3)],
                                                                      [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3],
                                                                       [0.4, 0.3, 0.3]], perturb=0.02)
        super().__init__(sampler)

class UniqueMotifHardNoColorWrapper(MotifWrapper):
    def __init__(self):
        sampler = UniqueMultipleOccurrencesMotifCategorizationDataset(BinaryTreeMotif(5, [0], 1),
                                                                      [HouseMotif([0], [0], 1, 1, 1),
                                                                       FullyConnectedMotif(5, [0], 1, 2)],
                                                                      [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
                                                                      perturb=0.02)
        super().__init__(sampler)


class UniqueMotifEasyWrapper(MotifWrapper):
    def __init__(self, num_colors=3):
        sampler = UniqueMultipleOccurrencesMotifCategorizationDataset(BinaryTreeMotif(5, [0], num_colors),
                                                                      [HouseMotif([1], [1], num_colors, 1, 1),
                                                                       FullyConnectedMotif(5, [2], num_colors, 2)],
                                                                      [[0.4, 0.6], [0.4, 0.6]])
        super().__init__(sampler)


__all__ = [MutagWrapper, EnzymesWrapper, RedditBinaryWrapper, UniqueMotifWrapper, UniqueMotifHardColorWrapper,
           UniqueMotifHardNoColorWrapper]


def from_name(name: str):
    for w in __all__:
        if w.__name__[:-7].lower() == name.replace("-", "").lower():
            return w()
    raise ValueError(f"Could not find wrapper matching dataset name {name}")
