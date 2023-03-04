from torch_geometric.datasets import TUDataset

datasets = {
    "MUTAG": TUDataset(root='/tmp', name='MUTAG'),
    "ENZYMES": TUDataset(root='/tmp', name='ENZYMES')
}