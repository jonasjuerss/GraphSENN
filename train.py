import argparse
import os.path
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from typing import Any
import re

import torch
import torch.nn.functional as F
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm

import custom_logger
import datasets
from custom_logger import log
from graph_senn import GraphSENN
from pooling_layers import StandardPoolingLayer, GraphSENNPool

CONV_TYPES = [GCNConv]

def train_test_epoch(train: bool, model: GraphSENN, optimizer, loader: DataLoader, epoch: int):
    if train:
        model.train()
    else:
        model.eval()
    correct = 0
    sum_loss = 0
    sum_class_loss = 0
    sum_reg_loss = 0
    num_classes = model.output_dim
    class_counts = torch.zeros(num_classes)
    with nullcontext() if train else torch.no_grad():
        for data in loader:
            data = data.to(custom_logger.device)
            batch_size = data.y.size(0)
            if train:
                optimizer.zero_grad()

            out, x_out, theta, h = model(data.x, data.edge_index, data.batch)
            target = data.y
            classification_loss = F.nll_loss(out, target)
            reg_loss = model.pooling_layer.calculate_stability_loss(model, data.x, data.batch, data.edge_index,
                                                                    theta, batch_size)
            loss = classification_loss + reg_loss

            sum_loss += batch_size * float(loss)
            sum_class_loss += batch_size * classification_loss
            # TODO change to more general additional_losses
            sum_reg_loss += batch_size * reg_loss

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
         f"{mode}_class_loss": sum_class_loss / dataset_len,
         f"{mode}_reg_loss": sum_reg_loss / dataset_len,
         f"{mode}_accuracy": correct / dataset_len, **distr_dict},
        step=epoch)

def main(args, **kwargs) -> tuple[GraphSENN, Any, DataLoader, DataLoader]:
    """
    :param args: The configuration as defined by the commandline arguments
    :param kwargs: additional kwargs to overwrite a loaded config with
    :return: The loaded/trained model, train and test data
    """
    if not isinstance(args, dict):
        args = args.__dict__

    if args["resume"] is not None:
        api = wandb.Api()
        run = api.run("jonas-juerss/graph-senn/" + args["resume"])
        save_path = args["save_path"]
        args = run.config
        restore_path = args["save_path"]
        args["save_path"] = save_path
        for k, v in kwargs.items():
            args[k] = v

    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    args = custom_logger.init(args)

    device = torch.device(args.device)
    custom_logger.device = device
    torch.manual_seed(args.seed)

    data_wrapper = datasets.from_name(args.dataset)
    dataset = data_wrapper.dataset
    num_classes = data_wrapper.num_classes
    num_node_features = data_wrapper.num_node_features
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
    gnn_activation = getattr(torch.nn, args.gnn_activation)

    gnn_output_size = args.gnn_sizes[-1] if args.gnn_sizes else num_node_features
    if args.senn_pooling:
        pool = GraphSENNPool(gnn_output_size, num_classes, args.theta_sizes, args.h_sizes, args.aggregation,
                             args.per_class_theta, args.per_class_h, args.global_theta, args.theta_loss_weight)
    else:
        pool = StandardPoolingLayer(gnn_output_size, num_classes, args.out_sizes, args.aggregation)


    model = GraphSENN(args.gnn_sizes, num_node_features, num_classes, conv_type, gnn_activation, pool)
    if args.resume is not None:
        model.load_state_dict(torch.load(restore_path))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    for epoch in tqdm(range(args.num_epochs)):
        train_test_epoch(True, model, optimizer, train_loader, epoch)
        if epoch % args.graph_log_freq == 0:
            pass
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), args.save_path)
        train_test_epoch(False, model, optimizer, test_loader, epoch)

    if args.graph_log_freq >= 0:
        pass
    return model, args, train_loader, test_loader


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
    parser.add_argument('--gnn_sizes', type=int, nargs='*',
                        default=[32, 32, 32, 32, 32], dest='gnn_sizes',
                        help='The layer sizes to use for the GNN.')
    parser.add_argument('--conv_type', type=str, default="GCNConv", choices=[c.__name__ for c in CONV_TYPES],
                        help='The type of graph convolution to use.')
    parser.add_argument('--gnn_activation', type=str, default="LeakyReLU",
                        help='Activation function to be used in between the GNN layers')
    parser.add_argument('--aggregation', type=str, default="Sum", choices=["Sum", "Mean", "Max", "Min", "Mul", "Var",
                                                                           "Std", "Softmax", "PowerMean"],
                        help='The aggregation function to use over all nodes in the output layer.')
    parser.add_argument('--senn_pooling', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to use our SENN pooling. Baseline otherwise.")

    # SENN
    parser.add_argument('--h_sizes', type=int, nargs='*',
                        default=[128, 1], dest='h_sizes',
                        help='The layer sizes to use for the h network. Can be empty for identity (of last embedding).')
    parser.add_argument('--per_class_h', default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to use a different concept scalar h per class or the same one for all.")
    parser.add_argument('--theta_sizes', type=int, nargs='*',
                        default=[128, 4], dest='theta_sizes',
                        help='The layer sizes to use for theta network. Can be empty for identity (of last embedding).')
    parser.add_argument('--per_class_theta', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to use a different concept weight theta per class (this is what SENN does) or the"
                             " same one for all.")
    parser.add_argument('--global_theta', default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to generate theta globally, i.e. concatenate a globally pooled embedding to the "
                             "node embedding when generating theta.")  # This reassembles an attention mechanism
    parser.add_argument('--theta_loss_weight', type=float, default=0.5,
                        help='The weight lambda of the theta regularization loss.')

    # No SENN
    parser.add_argument('--out_sizes', type=int, nargs='*',
                        default=[128, 128, 2], dest='out_sizes',
                        help='The layer sizes to use for the network after aggregation when not using SENN.')


    # Dataset
    parser.add_argument('--dataset', type=str, default="MUTAG", choices=[re.sub(r'(?<!^)(?=[A-Z])', '-', d.__name__[:-7]).\
                        upper() for d in datasets.__all__],
                        help='The name of the dataset to use as defined in datasets.py')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='The part of the dataset to use as train set (in (0, 1)).')


    # Setup
    parser.add_argument('--device', type=str, default="cuda",
                        help='The device to train on. Allows to use CPU or different GPUs.')
    parser.add_argument('--seed', type=int, default=1,
                        help='The seed used for pytorch. This also determines the dataset if generated randomly.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Will load configuration from the given wandb run and load the locally stored weights.')


    # Logging
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Every how many epochs to save the model. Set to -1 to disable saving to a file. The '
                             'last checkpoint will always be overwritten with the current one.')
    parser.add_argument('--save_path', type=str,
                        default=os.path.join("models", datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".pt"),
                        help='The path to save the checkpoint to. Will be models/dd-mm-YY_HH-MM-SS.pt by default.')
    parser.add_argument('--graph_log_freq', type=int, default=50,
                        help='Every how many epochs to log graphs to wandb. The final predictions will always be '
                             'logged, except for if this is negative.')
    parser.add_argument('--graphs_to_log', type=int, default=3,
                        help='How many graphs from the training and testing set to log.')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')

    args = parser.parse_args()
    main(args)

