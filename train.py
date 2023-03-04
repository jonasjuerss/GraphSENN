import argparse
import json
import typing
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm

import custom_logger
import datasets
from custom_logger import log
from graph_senn import GraphSENN
from pooling_layers import StandardPoolingLayer, GraphSENNPool

device = None
CONV_TYPES = [GCNConv]

def train_test_epoch(train: bool, model: GraphSENN, optimizer, loader: DataLoader, epoch: int):
    if train:
        model.train()
    else:
        model.eval()
    correct = 0
    sum_loss = 0
    num_classes = model.output_dim
    class_counts = torch.zeros(num_classes)
    with nullcontext() if train else torch.no_grad():
        for data in loader:
            data = data.to(device)
            batch_size = data.y.size(0)
            if train:
                optimizer.zero_grad()

            out = model(data)
            target = data.y
            classification_loss = F.nll_loss(out, target)
            loss = classification_loss  # + model.custom_losses(batch_size)

            sum_loss += batch_size * float(loss)
            pred_classes = out.argmax(dim=1)
            correct += int((pred_classes == target).sum())
            class_counts += torch.bincount(pred_classes.detach(), minlength=num_classes).cpu()

            if train:
                loss.backward()
                optimizer.step()
    dataset_len = len(loader.dataset)
    mode = "train" if train else "test"
    distr_dict = {}
    class_counts /= dataset_len
    if not train:
        distr_dict = {f"{mode}_percentage_class_{i}": class_counts[i] for i in range(num_classes)}
    log({f"{mode}_loss": sum_loss / dataset_len,
         f"{mode}_accuracy": correct / dataset_len, **distr_dict},
        step=epoch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Details
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The Adam learning rate to use.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='The Adam weight decay to use.')
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use.')


    # Architecture
    parser.add_argument('--gnn_sizes', type=int, nargs='+',
                        default=[32, 32, 32, 32, 32], dest='gnn_sizes',
                        help='The layer sizes to use for the GNN.')
    parser.add_argument('--conv_type', type=str, default="GCNConv", choices=[c.__name__ for c in CONV_TYPES],
                        help='The type of graph convolution to use.')
    parser.add_argument('--gnn_activation', type=str, default="leaky_relu",
                        help='Activation function to be used in between the GNN layers')
    parser.add_argument('--aggregation', type=str, default="Sum", choices=["Sum", "Mean", "Max", "Min", "Mul", "Var",
                                                                           "Std", "Softmax", "PowerMean"],
                        help='The aggregation function to use over all nodes in the output layer.')
    parser.add_argument('--senn_pooling', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to use our SENN pooling. Baseline otherwise.")

    # SENN
    parser.add_argument('--h_sizes', type=int, nargs='*',
                        default=[128, 2], dest='h_sizes',
                        help='The layer sizes to use for the h network. Can be empty for identity (of last embedding).')
    parser.add_argument('--per_class_h', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to use a different concept scalar h per class or the same one for all.")
    parser.add_argument('--theta_sizes', type=int, nargs='*',
                        default=[128, 2], dest='theta_sizes',
                        help='The layer sizes to use for theta network. Can be empty for identity (of last embedding).')
    parser.add_argument('--per_class_theta', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to use a different concept weight theta per class (this is what SENN does) or the"
                             " same one for all.")
    parser.add_argument('--global_theta', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to generate theta globally, i.e. concatenate a globally pooled embedding to the "
                             "node embedding when generating theta.")  # This ressembles an attention mechanism

    # No SENN
    parser.add_argument('--out_sizes', type=int, nargs='*',
                        default=[128, 128, 2], dest='out_sizes',
                        help='The layer sizes to use for the network after aggregation when not using SENN.')


    # Dataset
    parser.add_argument('--dataset', type=str, default="MUTAG", choices=datasets.datasets.keys(),
                        help='The name of the dataset to use as defined in datasets.py')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='The part of the dataset to use as train set (in (0, 1)).')


    # Setup
    parser.add_argument('--device', type=str, default="cuda",
                        help='The device to train on. Allows to use CPU or different GPUs.')
    parser.add_argument('--seed', type=int, default=1,
                        help='The seed used for pytorch. This also determines the dataset if generated randomly.')


    # Logging
    parser.add_argument('--graph_log_freq', type=int, default=50,
                        help='Every how many epochs to log graphs to wandb. The final predictions will always be '
                             'logged, except for if this is negative.')
    parser.add_argument('--graphs_to_log', type=int, default=3,
                        help='How many graphs from the training and testing set to log.')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')


    args = parser.parse_args()
    args = custom_logger.init(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    dataset = datasets.datasets[args.dataset]
    num_train_samples = int(args.train_split * len(dataset))
    train_data = dataset[:num_train_samples]
    test_data = dataset[num_train_samples:]
    graphs_to_log = train_data[:args.graphs_to_log] + test_data[:args.graphs_to_log]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    log_graph_loader = DataLoader(graphs_to_log, batch_size=1, shuffle=False)

    conv_type = next((x for x in CONV_TYPES if x.__name__ == args.conv_type), None)
    if conv_type is None:
        raise ValueError(f"No convolution type named \"{args.conv_type}\" found!")
    gnn_activation = getattr(torch.nn.functional, args.gnn_activation)

    if args.senn_pooling:
        pool = GraphSENNPool(args.gnn_sizes[-1], dataset.num_classes, args.theta_sizes, args.h_sizes, args.aggregation,
                             args.per_class_theta, args.per_class_h, args.global_theta)
    else:
        pool = StandardPoolingLayer(args.gnn_sizes[-1], dataset.num_classes, args.out_sizes, args.aggregation)


    model = GraphSENN(args.gnn_sizes, dataset.num_node_features, dataset.num_classes, conv_type, gnn_activation, pool).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in tqdm(range(args.num_epochs)):
        train_test_epoch(True, model, optimizer, train_loader, epoch)
        if epoch % args.graph_log_freq == 0:
            pass
        train_test_epoch(False, model, optimizer, test_loader, epoch)

    if args.graph_log_freq >= 0:
        pass

