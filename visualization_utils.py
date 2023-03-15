import matplotlib
import networkx as nx
import numpy as np
import torch_geometric
from matplotlib import pylab, pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import k_hop_subgraph

import datasets
from datasets import DatasetWrapper
from graph_senn import GraphSENN
from train import main

class InterpretationData():
    def __init__(self, wandb_id: str, data_loader: DataLoader = None):
        train_loader, val_loader, test_loader = self.load_model(wandb_id)
        self.data_loader = test_loader if data_loader is None else data_loader
        self.load_whole_dataset()
        self.colors = np.array([matplotlib.colors.rgb2hex(c) for c in pylab.get_cmap("tab20").colors])
    def load_model(self, wandb_id: str):
        self.model, self.config, train_loader, val_loader, test_loader =\
            main(dict(resume=wandb_id, save_path="models/dummy.pt"), use_wandb=False, num_epochs=0, shuffle=False)
        self.model.eval()
        self.num_hops = len(self.config.gnn_sizes)
        self.device = torch.device(self.config.device)
        self.final_emb_dim = self.config.gnn_sizes[-1]
        self.dataset_wrapper = datasets.from_name(self.config.dataset)
        return train_loader, val_loader, test_loader

    def load_whole_dataset(self):
        h_all = torch.empty(0, 1, device=self.config.device)
        theta_all = torch.empty(0, self.model.output_dim, device=self.config.device)
        batch_all = torch.empty(0, device=self.config.device, dtype=torch.long)
        x_all = torch.empty(0, self.model.input_dim, device=self.config.device)
        x_out_all = torch.empty(0, self.final_emb_dim, device=self.config.device)
        y_all = torch.empty(0, device=self.config.device, dtype=torch.long)
        annot_all = torch.empty(0, device=self.config.device, dtype=torch.long)
        edge_index_all = torch.empty(2, 0, device=self.config.device, dtype=torch.long)
        out_all = torch.empty(0, self.model.output_dim, device=self.config.device)
        num_samples = 0
        num_nodes = 0
        with torch.no_grad():
            for data in self.data_loader:
                data.cuda()
                out, x_out, theta, h = self.model(data.x, data.edge_index, data.batch, data.annotations)
                h_all = torch.cat((h_all, h), dim=0)
                theta_all = torch.cat((theta_all, theta), dim=0)
                batch_all = torch.cat((batch_all, data.batch + num_samples), dim=0)
                x_all = torch.cat((x_all, data.x), dim=0)
                x_out_all = torch.cat((x_out_all, x_out), dim=0)
                y_all = torch.cat((y_all, data.y), dim=0)
                annot_all = torch.cat((annot_all, data.annotations), dim=0)
                edge_index_all = torch.cat((edge_index_all, data.edge_index + num_nodes), dim=1)
                out_all = torch.cat((out_all, out), dim=0)
                num_samples += len(data)
                num_nodes += data.x.shape[0]

        self.h_all = h_all.detach().cpu().numpy()
        self.theta_all = theta_all.detach().cpu().numpy()
        self.batch_all = batch_all.detach().cpu().numpy()
        self.x_all = x_all.detach().cpu().numpy()
        self.x_out_all = x_out_all.detach().cpu().numpy()
        self.y_all = y_all.detach().cpu().numpy()
        self.annot_all = annot_all.detach().cpu().numpy()
        self.y_all_nodes = y_all[batch_all]
        self.edge_index_all = edge_index_all.detach().cpu().numpy()
        self.y_pred_all = torch.argmax(out_all, dim=1).detach().cpu().numpy()
        self.out_all = out_all.detach().cpu().numpy()
        print(f"Accuracy: {100 * np.sum(self.y_pred_all == self.y_all) / self.y_all.shape[0]:.2f}%")

    def plot_clusters(self, K: int):
        # [num_nodes_total, x, y] (PCA or t-SNE)
        coords = TSNE(n_components=2).fit_transform(X=self.x_out_all)
        kmeans = KMeans(n_clusters=K).fit(X=self.x_out_all)
        # [num_nodes_total] with integer values between 0 and K/num_clusters/num_concepts
        clusters = kmeans.labels_

        markers = ["o", "p", "s", "P", "*", "D", "^", "+", "x"]

        fig, ax = plt.subplots()
        for i in range(self.model.output_dim):
            # Note, we could also evaluate how the model classifies (y_pred_all) instead of the ground truth (y_all) for explainability
            ax.scatter(coords[self.y_all_nodes == i, 0], coords[self.y_all_nodes == i, 1],
                       c=self.colors[clusters[self.y_all_nodes == i]],
                       marker=markers[i], s=10)

    def draw_neighbourhood(self, ax, node_index: int):
        subset, edge_index, mapping, _ = k_hop_subgraph(node_index, self.num_hops, torch.tensor(self.edge_index_all),
                                                        relabel_nodes=True)
        g = torch_geometric.utils.to_networkx(Data(x=subset, edge_index=edge_index), to_undirected=True)
        highlight_mask = np.zeros(subset.shape[0], dtype=int)
        highlight_mask[mapping] = 1
        labels = self.dataset_wrapper.get_node_labels(self.x_all[subset])
        nx.draw(g, ax=ax, node_color=self.dataset_wrapper.get_node_colors(self.annot_all[subset]),
                edgecolors="#8e44ad",
                linewidths=3.0*highlight_mask, labels={i: labels[i] for i in range(labels.shape[0])},
                font_color="whitesmoke")

    def plot_closest_embeddings(self, embeddings: np.ndarray, labels, save_path=None, num_plots = 5):
        """
        For each row in embeddings, plots the num_gnn_layers-hop neighbourhood of the num_plots closest embeddings in x_out_all
        :param embeddings: [num_embeddings, gnn_output_embedding_size]
        """
        fig, axes = plt.subplots(embeddings.shape[0], num_plots, figsize=(15, embeddings.shape[0] * 5))
        for i in range(embeddings.shape[0]):
            # CAUTION: we are using TOP k, so we need a minus in front to find the ones with smallest distance
            # indices of the num_plots nodes that are closest to the center of concept
            _, indices = torch.topk(-torch.norm(torch.tensor(embeddings[i:i+1, :] - self.x_out_all), dim=-1), k=num_plots, dim=0)
            axes[i][0].set_title(f"{labels[i]}", rotation='vertical', x=-0.1, y=0.5)
            for j in range(num_plots):
                self.draw_neighbourhood(axes[i][j], indices[j].item())
        if save_path is not None:
            fig.savefig(save_path)
        return fig

    def draw_graph(self, sample: int, save_path=None, figsize=(7, 7)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        mask = self.batch_all == sample
        node_indices = np.arange(self.x_all.shape[0])[mask]
        start_index, end_index = node_indices[0].item(), node_indices[-1].item()
        edge_index = self.edge_index_all[:,
                     np.logical_and(self.edge_index_all[0] >= start_index, self.edge_index_all[0] <= end_index)] - start_index
        x = self.x_all[mask]
        g = torch_geometric.utils.to_networkx(Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index)),
                                              to_undirected=True)
        labels = self.dataset_wrapper.get_node_labels(x)
        pred_str = ", ".join([f"{100 * f:.0f}%" for f in np.exp(self.out_all[sample])])
        ax.set_title(
            f"class: {self.y_all[sample]} ({self.dataset_wrapper.class_names[self.y_all[sample].item()]}), prediction: [{pred_str}]")
        nx.draw(g, ax=ax, node_color=self.dataset_wrapper.get_node_colors(self.annot_all[mask]),
                labels={i: str(i) for i in range(labels.shape[0])}, font_color="whitesmoke")

        if save_path is not None:
            fig.savefig(save_path)
        return fig

    def plot_theta_and_h(self, sample: int, save_path=None):
        fig, axes = plt.subplots(2, self.model.output_dim, figsize=(15,5))
        fig.tight_layout()
        mask = self.batch_all==sample
        theta = self.theta_all[mask]
        h = self.h_all[mask]
        colors = self.dataset_wrapper.get_node_colors(self.annot_all[mask])
        axes[0][0].set_ylabel("node/concept id $i$")
        axes[1][0].set_ylabel("node/concept id $i$")
        for i in range(self.model.output_dim):
            axes[0][i].set_title(self.dataset_wrapper.class_names[i])
            axes[0][i].barh(np.arange(theta.shape[0]), theta[:, 0 if theta.shape[1] == 1 else i], color=colors)
            axes[0][i].set_xlabel(f"$\\theta_i(x)$")
            axes[0][i].invert_yaxis()
            axes[1][i].barh(np.arange(h.shape[0]), h[:, 0 if h.shape[1] == 1 else i], color=colors)
            axes[1][i].set_xlabel(f"$h_i(x)$")
            axes[1][i].invert_yaxis()
        if save_path is not None:
            fig.savefig(save_path)
        return fig

    # Show neighbourhoods of nodes with the most similar embeddings
    def show_nearest(self, sample: int, num_samples_per_concept=5, save_path=None):
        return self.plot_closest_embeddings(self.x_out_all[self.batch_all == sample, :],
                                            [f"Node {i}" for i in range(self.x_out_all[self.batch_all == sample, :].shape[0])],
                                            num_plots=num_samples_per_concept, save_path=save_path)

    # Show neighbourhoods of nodes whos embeddings are closest to the desired one-hot vector
    def show_nearest_discretized(self, sample: int, num_samples_per_concept=5, save_path=None):
        max_indices = np.argmax(self.x_out_all[self.batch_all == sample, :], axis=-1)
        # [num_present_concepts, gnn_sizes[-1]]
        present_concepts = np.eye(self.x_out_all.shape[1])[np.unique(max_indices), :]
        labels = [None for _ in range(self.x_out_all.shape[1])]
        for i in range(max_indices.shape[0]):
            labels[max_indices[i]] = (f"Concept {max_indices[i]}, Nodes:\n" if labels[max_indices[i]] is None else
                                      labels[max_indices[i]] + ", ") + str(i)

        return self.plot_closest_embeddings(present_concepts, [l for l in labels if l is not None],
                                            num_plots=num_samples_per_concept, save_path=save_path)

    def show_category_histogram(self, save_path=None):
        counts = np.bincount(np.argmax(self.x_out_all, axis=1))
        fig, ax = plt.subplots()
        ax.bar(np.arange(counts.shape[0]), counts)
        if save_path is not None:
            fig.savefig(save_path)
        return fig