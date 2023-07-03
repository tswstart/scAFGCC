# -*- coding: utf-8 -*-
# @Author  : sw t
# @Time    : 2023/4/12 21:02
import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset
import warnings
import scanpy as sc
import numpy as np
from graph import get_adj
from torch_geometric.utils import remove_self_loops, to_undirected

class scRNAGraph(InMemoryDataset):

    def __init__(self, root, name, input_h5ad_path, transform=None, pre_transform=None, is_undirected=None):
        """
        @param root: root path to save file
        @param name: name of datasets
        @param path: path of input h5ad file
        @param transform:
        @param pre_transform:
        """
        self.name = name
        self.input_h5ad_path = input_h5ad_path
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default.")
            is_undirected = True
        self.is_undirected = is_undirected
        super(scRNAGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return 'data_raw.pt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        adata = sc.read_h5ad(self.input_h5ad_path)
        if isinstance(adata.X, np.ndarray):
            x = adata.X
        else:
            x = adata.X.toarray()

        if adata.obs.get('x') is not None:
            y = adata.obs['x'].values
            cell_type, y = np.unique(y, return_inverse=True)
        else:
            y = None
            print("Can not find corresponding labels")
        adj = get_adj(x)
        edge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, x.shape[0])
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
if __name__ == '__main__':
    _ = scRNAGraph('data', 'datasetname', 'h5ad/real_data/yan_preprocessed.h5ad')