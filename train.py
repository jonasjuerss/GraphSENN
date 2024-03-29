import argparse
import os.path
import traceback
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Tuple
import re

import torch
import torch.nn.functional as F
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from tqdm import tqdm

import custom_logger
import datasets
from custom_logger import log
from decoders import AdjGenerationType, FullyConnectedMessagePassingDecoder
from graph_senn import GraphSENN
from pooling_layers import StandardPoolingLayer, GraphSENNPool

CONV_TYPES = [GCNConv, GATConv]

def train_test_epoch(train: bool, model: GraphSENN, optimizer, loader: DataLoader, epoch: int, mode_str: str):
    if train:
        model.train()
    else:
        model.eval()
    correct = 0
    sum_loss = 0
    sum_class_loss = 0
    sum_reg_loss = 0
    sum_add_loss = {}
    num_classes = model.output_dim
    class_counts = torch.zeros(num_classes)
    with nullcontext() if train else torch.no_grad():
        for data in loader:
            data = data.to(custom_logger.device)
            batch_size = data.y.size(0)
            if train:
                optimizer.zero_grad()

            annotations = data.annotations if hasattr(data, "annotations") else None
            out, x_out, theta, h = model(data.x, data.edge_index, data.batch, annotations)
            target = data.y
            classification_loss = F.nll_loss(out, target)
            reg_loss, add_loss_dict = model.pooling_layer.calculate_additional_losses(model, data.x, x_out, data.batch,
                                                                                      data.edge_index, theta, h,
                                                                                      batch_size)

            loss = classification_loss + reg_loss
            if torch.isnan(loss).item():
                raise ValueError("NaN encountered during training!")

            sum_loss += batch_size * float(loss)
            sum_class_loss += batch_size * classification_loss
            sum_reg_loss += batch_size * reg_loss
            for k, v in add_loss_dict.items():
                sum_add_loss[k] = sum_add_loss.get(k, 0) + batch_size * v

            pred_classes = out.argmax(dim=1)
            correct += int((pred_classes == target).sum())
            class_counts += torch.bincount(pred_classes.detach(), minlength=num_classes).cpu()

            if train:
                loss.backward()
                optimizer.step()
    dataset_len = len(loader.dataset)
    distr_dict = {}
    class_counts /= dataset_len
    if mode_str == "test":
        distr_dict = {f"{mode_str}_percentage_class_{i}": class_counts[i] for i in range(num_classes)}
    res_dict = {
        f"{mode_str}_loss": sum_loss / dataset_len,
        f"{mode_str}_class_loss": sum_class_loss / dataset_len,
        f"{mode_str}_reg_loss": sum_reg_loss / dataset_len,
        f"{mode_str}_accuracy": correct / dataset_len,
        **{f"{mode_str}_{k}": v / dataset_len for k, v in sum_add_loss.items()},
        **distr_dict}
    log(res_dict, step=epoch)
    return res_dict

def main(args, **kwargs) -> Tuple[GraphSENN, Any, DataLoader, DataLoader, DataLoader]:
    """
    :param args: The configuration as defined by the commandline arguments
    :param kwargs: additional kwargs to overwrite a loaded config with
    :return: The loaded/trained model, train and test data
    """
    if not isinstance(args, dict):
        args = args.__dict__
    restore_path = None
    if args["resume"] is not None:
        api = wandb.Api()
        run_path = f"{custom_logger.wandb_entity}/{custom_logger.wandb_project}/" + args["resume"]
        run = api.run(run_path)
        save_path = args["save_path"]
        args = run.config
        restore_path = args["save_path"]
        if args["save_wandb"] and not os.path.isfile(restore_path):
            print("Downloading checkpoint from wandb...")
            wandb.restore(restore_path, run_path=run_path)
        args["save_path"] = save_path
        for k, v in kwargs.items():
            args[k] = v
    else:
        if not args["use_wandb"] and args["save_wandb"]:
            print("Disabling saving to wandb as logging to wandb is also disabled.")
            args["save_wandb"] = False

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
    num_val_samples = int(args.val_split * len(dataset))
    train_data = dataset[:num_train_samples]
    val_data = dataset[num_train_samples:num_train_samples+num_val_samples]
    test_data = dataset[num_train_samples + num_val_samples:]
    graphs_to_log = train_data[:args.graphs_to_log] + test_data[:args.graphs_to_log]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=kwargs.get("shuffle", True))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=kwargs.get("shuffle", True))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=kwargs.get("shuffle", True))
    log_graph_loader = DataLoader(graphs_to_log, batch_size=1, shuffle=False)

    conv_type = next((x for x in CONV_TYPES if x.__name__ == args.conv_type), None)
    if conv_type is None:
        raise ValueError(f"No convolution type named \"{args.conv_type}\" found!")
    gnn_activation = getattr(torch.nn, args.gnn_activation)

    gnn_output_size = args.gnn_sizes[-1] if args.gnn_sizes else num_node_features
    if args.senn_pooling:
        if args.feat_reconst_loss_weight != 0 or args.adj_reconst_loss_weight != 0:
            decoder = FullyConnectedMessagePassingDecoder(args.gnn_sizes, num_node_features, "SAGEConv", gnn_activation,
                                                          args.h_adj_dec_intermediate, args.h_adj_dec_final)
        else:
            decoder = None
        pool = GraphSENNPool(gnn_output_size, num_classes, args.theta_sizes, args.h_sizes, args.aggregation,
                             args.per_class_theta, args.per_class_h, args.global_theta, args.theta_loss_weight,
                             args.feat_reconst_loss_weight, args.adj_reconst_loss_weight, decoder, True)  # args.learn_h)
    else:
        pool = StandardPoolingLayer(gnn_output_size, num_classes, args.out_sizes, args.aggregation)


    model = GraphSENN(args.gnn_sizes, num_node_features, num_classes, conv_type, gnn_activation, pool,
                      args.concept_activation)
    if restore_path is not None:
        model.load_state_dict(torch.load(restore_path))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_val_acc = 0
    try:
        for epoch in tqdm(range(args.num_epochs)):
            train_test_epoch(True, model, optimizer, train_loader, epoch, "train")
            train_test_epoch(False, model, optimizer, val_loader, epoch, "test")
            val_acc = train_test_epoch(False, model, optimizer, test_loader, epoch, "val")["val_accuracy"]
            if epoch % args.graph_log_freq == 0:
                pass
            if (args.save_freq > 0 and epoch % args.save_freq == 0) or\
                    (args.save_freq == -2 and val_acc > best_val_acc):
                torch.save(model.state_dict(), args.save_path)
                if args.save_wandb:
                    wandb.save(args.save_path, policy="now")
                if args.save_freq == -2:
                    print(f"Validation accuracy {100 * val_acc:.2f}%. Saving.")
            best_val_acc = max(val_acc, best_val_acc)
        if args.num_epochs > 0:
            log({"best_val_acc": best_val_acc}, step=epoch)
    except:
        log({"best_val_acc": -1}, step=epoch)
        traceback.print_exc()

    return model, args, train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Details
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The Adam learning rate to use.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='The Adam weight decay to use.')
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use.')

    # Architecture
    parser.add_argument('--gnn_sizes', type=int, nargs='*',
                        default=[32, 32, 32], dest='gnn_sizes',
                        help='The layer sizes to use for the GNN.')
    parser.add_argument('--conv_type', type=str, default="GCNConv", choices=[c.__name__ for c in CONV_TYPES],
                        help='The type of graph convolution to use. Note: GATConv does not appear to work with h loss.')
    parser.add_argument('--gnn_activation', type=str, default="LeakyReLU",
                        help='Activation function to be used in between the GNN layers')
    parser.add_argument('--aggregation', type=str, default="Sum", choices=["Sum", "Mean", "Max", "Min", "Mul", "Var",
                                                                           "Std", "Softmax", "PowerMean"],
                        help='The aggregation function to use over all nodes in the output layer.')
    parser.add_argument('--senn_pooling', action='store_true', help="Whether to use our SENN pooling. Baseline "
                                                                    "otherwise.")
    parser.add_argument('--no-senn_pooling', dest='senn_pooling', action='store_false')
    parser.set_defaults(senn_pooling=True)

    # SENN

    parser.add_argument('--concept_activation', type=str, default="none",
                        choices=["none", "sigmoid", "softmax", "gumbel_softmax", "gumbel_softmax_soft"],
                        help='The function applied to the last node embeddings before they serve as input to the h/'
                             'theta networks.')

    # h
    parser.add_argument('--h_sizes', type=int, nargs='*',
                        default=[128, 1], dest='h_sizes',
                        help='The layer sizes to use for the h network. Can be empty for identity (of last embedding).')
    parser.add_argument('--per_class_h', action='store_true', help="Whether to use a different concept scalar h per "
                                                                   "class or the same one for all.")
    parser.add_argument('--no-per_class_h', dest='per_class_h', action='store_false')
    parser.set_defaults(per_class_h=False)
    parser.add_argument('--feat_reconst_loss_weight', type=float, default=0,
                        help='The weight of the feature reconstruction in the reconstruction loss.')
    parser.add_argument('--adj_reconst_loss_weight', type=float, default=0,
                        help='The weight of the adjacency reconstruction in the reconstruction loss.')
    parser.add_argument('--h_adj_dec_intermediate', type=str, default=AdjGenerationType.IDENTITY.value,
                        choices=[v.value for v in AdjGenerationType.__members__.values()],
                        help='The type of adjacency reconstruction in intermediate layers when using the '
                             'FullyConnectedMessagePassingDecoder for the h loss.')
    parser.add_argument('--h_adj_dec_final', type=str, default=AdjGenerationType.MLP.value,
                        choices=[v.value for v in AdjGenerationType.__members__.values()],
                        help='The type of adjacency reconstruction for the final output when using the '
                             'FullyConnectedMessagePassingDecoder for the h loss.')
    parser.add_argument('--learn_h', action='store_true', help="Whether to learn h from the GNN output. Otherwise will "
                                                               "use one-hot vector of ground truth annotations "
                                                               "(assuming they are present and fit)")
    parser.add_argument('--no-learn_h', dest='learn_h', action='store_false')
    parser.set_defaults(learn_h=True)

    # Theta
    parser.add_argument('--theta_sizes', type=int, nargs='*',
                        default=[128, 4], dest='theta_sizes',
                        help='The layer sizes to use for theta network. Can be empty for identity (of last embedding).')
    parser.add_argument('--per_class_theta', action='store_true', help="Whether to use a different concept weight theta"
                                                                       " per class (this is what SENN does) or the same"
                                                                       " one for all.")
    parser.add_argument('--no-per_class_theta', dest='per_class_theta', action='store_false')
    parser.set_defaults(per_class_theta=True)
    parser.add_argument('--global_theta', action='store_true', help="Whether to generate theta globally, i.e. "
                                                                    "concatenate a globally pooled embedding to the "
                                                                    "node embedding when generating theta.")
    parser.add_argument('--no-global_theta', dest='global_theta', action='store_false')
    parser.set_defaults(global_theta=False) # This reassembles an attention mechanism
    parser.add_argument('--theta_loss_weight', type=float, default=0,
                        help='The weight lambda of the theta regularization loss.')

    # No SENN
    parser.add_argument('--out_sizes', type=int, nargs='*',
                        default=[128, 4], dest='out_sizes',
                        help='The layer sizes to use for the network after aggregation when not using SENN.')


    # Dataset
    parser.add_argument('--dataset', type=str, default="UNIQUE-MOTIF", choices=[re.sub(r'(?<!^)(?=[A-Z])', '-', d.__name__[:-7]).\
                        upper() for d in datasets.__all__],
                        help='The name of the dataset to use as defined in datasets.py')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='The part of the dataset to use as train set (in (0, 1)).')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='The part of the dataset to use as validation set (in (0, 1)).')


    # Setup
    parser.add_argument('--device', type=str, default="cuda",
                        help='The device to train on. Allows to use CPU or different GPUs.')
    parser.add_argument('--seed', type=int, default=1,
                        help='The seed used for pytorch. This also determines the dataset if generated randomly.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Will load configuration from the given wandb run and load the locally stored weights.')


    # Logging
    parser.add_argument('--save_freq', type=int, default=-2,
                        help='Every how many epochs to save the model. Set to -1 to disable saving to a file and to -2 '
                             'to save whenever validation accuracy improved. The  last checkpoint will always be '
                             'overwritten with the current one.')
    parser.add_argument('--save_path', type=str,
                        default="models/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".pt",
                        help='The path to save the checkpoint to. Will be models/dd-mm-YY_HH-MM-SS.pt by default.')
    parser.add_argument('--save_wandb', action='store_true', help="Whether to upload the checkpoint files to wandb!")
    parser.add_argument('--no-save_wandb', dest='save_wandb', action='store_false')
    parser.set_defaults(save_wandb=True)
    parser.add_argument('--graph_log_freq', type=int, default=50,
                        help='Every how many epochs to log graphs to wandb. The final predictions will always be '
                             'logged, except for if this is negative.')
    parser.add_argument('--graphs_to_log', type=int, default=3,
                        help='How many graphs from the training and testing set to log.')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')
    parser.add_argument('--wandb_name', type=str,
                        default=None,
                        help="Name of the wandb run. Standard randomly generated wandb names if not specified.")

    args = parser.parse_args()
    main(args)

