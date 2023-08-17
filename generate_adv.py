import numpy as np
import argparse
import os.path as osp
import pickle as pkl
from scipy.sparse import csr_matrix

import torch
from BlackBoxAttack.attack.PGD import FPGDAttack

from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.utils import preprocess
from datasets.common import get_node_dataset
import utils.process as process
import os


def attack_model(adj, features, labels, device):
    fmodel = None

    if args.rate < 1:
        n_perturbation = int(args.rate * dataset.data.num_edges / 2)
    else:
        n_perturbation = int(args.rate)

    n_candidates = n_perturbation * 10
    adj, features, labels = preprocess(adj, csr_matrix(features), labels, preprocess_adj=False)  # conver to tensor
    model = PGDAttack(surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
    if n_perturbation > 1e-7:
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbation)
    else:
        model.modified_adj = adj
    fmodel = FPGDAttack(surrogate, adj.shape[0], loss_type='CE', device=device).to(device)
    if args.eps_x > 1e-7:
        fmodel.attack(features, adj, labels, idx_train, eps_x=args.eps_x,
                      show_attack=False)
    else:
        fmodel.modified_features = features

    return model, fmodel


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='The dataset required.')
parser.add_argument('--rate', type=float, default=0.20, help='The ratio for edge perturbations.')
parser.add_argument('--eps_x', type=float, default=0.0010, help='The ratio for feature perturbations.')
parser.add_argument('--device', default='cpu', help='The devive you use.')

args = parser.parse_args()
device = args.device
device = torch.device(f'cuda:{device}')
path = osp.expanduser('dataset')
path = osp.join(path, args.dataset)
dataset, train_mask, val_mask, test_mask = get_node_dataset(args.dataset)
dataset.data.x, _ = process.preprocess_features_tensor(dataset.data.x.cpu())

mapping = None

data = Pyg2Dpr(dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, verbose=True)

# Setup Attack Model
model, fmodel = attack_model(adj, features, labels, device)
modified_adj = model.modified_adj

if not os.path.exists('poisoned_adj'):
    os.makedirs('poisoned_adj')

pkl.dump(modified_adj,
         open('poisoned_adj/%s_pgd_%f_%f_adj.pkl' % (args.dataset, args.rate, args.eps_x), 'wb'))

if fmodel is None:
    modified_features = features
else:
    modeified_features = fmodel.modified_features
    pkl.dump(modeified_features,
             open('poisoned_adj/%s_pgd_%f_%f_features.pkl' % (args.dataset, args.rate, args.eps_x), 'wb'))
