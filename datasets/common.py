import torch
from torch_geometric.datasets import Planetoid, CitationFull
from sklearn.model_selection import train_test_split
import numpy as np


def get_node_dataset(name, root='./node_dataset/', sparse=True, train_ratio=0.1, test_ratio=0.8):
    if name in ['cora', 'citeseer', 'pubmed']:
        full_dataset = Planetoid(root, name)
        train_mask = full_dataset[0].train_mask
        val_mask = full_dataset[0].val_mask
        test_mask = full_dataset[0].test_mask
        return full_dataset, train_mask, val_mask, test_mask
    elif name in ['cora_ml']:
        full_dataset = CitationFull(root, name)
        num_nodes = full_dataset.data.x.shape[0]
        idx = np.arange(0, num_nodes)
        labels = full_dataset.data.y.numpy()
        idx_train, idx_test = train_test_split(idx, test_size=test_ratio, random_state=1234, stratify=labels)
        idx_train, idx_val = train_test_split(idx_train, test_size=0.5, random_state=123, stratify=labels[idx_train])
        train_mask = np.zeros(num_nodes)
        val_mask = np.zeros(num_nodes)
        test_mask = np.zeros(num_nodes)
        train_mask[idx_train] = 1
        val_mask[idx_val] = 1
        test_mask[idx_test] = 1

        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)

        full_dataset.data.train_mask = train_mask
        full_dataset.data.val_mask = val_mask
        full_dataset.data.test_mask = test_mask

        return full_dataset, train_mask, val_mask, test_mask
    else:
        raise NotImplementedError
