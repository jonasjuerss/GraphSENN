import abc

import numpy as np
from torch_geometric.datasets import TUDataset


class DatasetWrapper(abc.ABC):
    def __init__(self, dataset):
        self.dataset = dataset

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
        super().__init__(TUDataset(root='/tmp', name='MUTAG'))
        self.label_map = np.array(['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca'])
        self.color_map = np.array(['#2c3e50', '#e74c3c', '#27ae60', '#3498db', '#CDDC39', '#f39c12', '#795548', '#8e44ad', '#3F51B5', '#7f8c8d', '#e84393', '#607D8B', '#8e44ad', '#009688'])

    def get_node_labels(self, x: np.ndarray) -> np.ndarray:
        return self.label_map[np.argmax(x, axis=-1)]

    def get_node_colors(self, x: np.ndarray) -> np.ndarray:
        return self.color_map[np.argmax(x, axis=-1)]


class EnzymesWrapper(DatasetWrapper):
    def __init__(self):
        super().__init__(TUDataset(root='/tmp', name='ENZYMES'))


__all__ = [MutagWrapper, EnzymesWrapper]


def from_name(name: str):
    for w in __all__:
        if w.__name__[:-7].lower() == name.lower():
            return w()
    raise ValueError(f"Could not find wrapper matching dataset name {name}")
