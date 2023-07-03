import numpy as np

np.random.seed(0)
import torch
import torch.nn as nn
from utils import printConfig
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random

random.seed(0)

import os

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, pairwise
import umap
import hdbscan


class embedder:
    def __init__(self, args):
        self.args = args
        self.hidden_layers = eval(args.layers)
        printConfig(args)

    def infer_embeddings(self, epoch):
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None
        # one batch
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            emb, _, _, _ = self._model(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                       neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                       edge_weight=batch_data.edge_attr, epoch=epoch)
            emb = emb.detach()
            y = batch_data.y.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])

    def evaluate(self, task, epoch):
        if task == "clustering":
            self.evaluate_clustering(epoch)

    def evaluate_clustering(self, epoch, clusters=''):

        embeddings = F.normalize(self._embeddings, dim=-1, p=2).detach().cpu().numpy()
        # self._dataset[0] 表示data图数据
        nb_class = len(self._dataset[0].y.unique())
        true_y = self._dataset[0].y.detach().cpu().numpy()
        if clusters == 'kmeans':
            estimator = KMeans(n_clusters = nb_class)

            ARI_list = []
            NMI_list = []

            for i in range(10):
                estimator.fit(embeddings)
                y_pred = estimator.predict(embeddings)

                s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
                s2 = adjusted_rand_score(true_y, y_pred)
                NMI_list.append(s1)
                ARI_list.append(s2)

            s1 = sum(NMI_list) / len(NMI_list)
            s2 = sum(ARI_list) / len(ARI_list)
            print('kmeans')
        else:
            umap_reducer = umap.UMAP()
            u = umap_reducer.fit_transform(embeddings)
            cl_sizes = [10, 25, 50, 100]
            min_samples = [5, 10, 25, 50]
            hdbscan_dict = {}
            ari_dict = {}
            for cl_size in cl_sizes:
                for min_sample in min_samples:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_size, min_samples=min_sample)
                    clusterer.fit(u)
                    y_pred = clusterer.labels_
                    nmi = normalized_mutual_info_score(true_y, y_pred)
                    ari = adjusted_rand_score(true_y, y_pred)
                    ari_dict[(cl_size, min_sample)] = {'NMI': nmi, 'ARI': ari}
                    hdbscan_dict[(cl_size, min_sample)] = y_pred
            max_tuple = max(ari_dict, key=lambda x: ari_dict[x]['ARI'])
            s2 = ari_dict[max_tuple]['ARI']
            s1 = ari_dict[max_tuple]['NMI']
            print('hdbscan')

        print('** [{}] [Current Epoch {}] Clustering ARI: {:.4f} NMI: {:.4f} **'.format(self.args.embedder, epoch, s2,
                                                                                        s1))

        if s2 > self.best_dev_acc_ari:
            self.best_epoch = epoch
            self.best_dev_acc_ari = s2
            self.best_dev_acc_nmi = s1
            if self._args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                torch.save(self._embeddings.detach().cpu(), os.path.join(self._args.checkpoint_dir,
                                                                         'embeddings_{}_{}.pt'.format(
                                                                             self._args.dataset.split('/')[2],
                                                                             self._args.task)))

        self.st_best = '** [Best epoch: {}] Best ARI: {:.4f} NMI: {:.4f} **\n'.format(self.best_epoch,
                                                                                      self.best_dev_acc_ari,
                                                                                      self.best_dev_acc_nmi)
        print(self.st_best)


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        self.stacked_gnn = nn.ModuleList(
            [GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)

        return x
