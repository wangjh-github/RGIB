import os

import torch
from datasets.common import get_node_dataset
from baselines import RGIB, eval_clf
import argparse

import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix

import utils.process as process
import pickle as pkl
import yaml
from yaml import SafeLoader

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--dataset', type=str, default='cora', help='The required dataset')
parser.add_argument('--emb_dim', type=int, default=512, help='The embedding dim of the encoder')
parser.add_argument('--nlayers', type=int, default=2, help='The number of layers of the GNN')
parser.add_argument('--training', action='store_true', dest='training',
                    help='Whether train the encoder or load exising encoder')
parser.set_defaults(training=False)

parser.add_argument('--adv', action='store_true', dest='adv',
                    help='Whether evaluate the representations on the adversarial graph')
parser.set_defaults(adv=False)

parser.add_argument('--rate', type=float, default=0.2, help='The edge attack ratio')
parser.add_argument('--eps_x', type=float, default=0.001, help='The features attack ratio')
parser.add_argument('--alpha', type=float, default=0.9,
                    help='The hyper-parameter to control the weight of the adversarial term')
parser.add_argument('--beta', type=float, default=0.1,
                    help='The hyper-parameter to control the weight of the KL divergence term')

parser.add_argument('--device', type=int, default=0, help='The device you use')
args = parser.parse_args()

dataset_name = args.dataset
embed_dim = args.emb_dim
dataset, train_mask, val_mask, test_mask = get_node_dataset(dataset_name)

config_path = f'configs/RGIB/config.yaml'

if os.path.exists(config_path):
    config = yaml.load(open(config_path), Loader=SafeLoader)[dataset_name]
    args.nlayers = config['nlayers']

if args.adv:
    adj = pkl.load(open('poisoned_adj/%s_pgd_%f_%f_adj.pkl' % (dataset_name, args.rate, args.eps_x), 'rb'))
    row, col = torch.nonzero(adj, as_tuple=True)
    edge_index = torch.stack((row, col), dim=0)
    dataset.data.edge_index = edge_index

    features = pkl.load(
        open('poisoned_adj/%s_pgd_%f_%f_features.pkl' % (dataset_name, args.rate, args.eps_x), 'rb'))
    dataset.data.x = features.detach()

adj = to_scipy_sparse_matrix(edge_index=dataset.data.edge_index)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.device)

train_mask = train_mask.cuda()
val_mask = val_mask.cuda()
test_mask = test_mask.cuda()

features = dataset.data.x
labels = dataset.data.y.cuda()

features, _ = process.preprocess_features_tensor(features)
adj = sp.coo_matrix(adj)

model = RGIB(dataset[0].x.shape[1], embed_dim, 'prelu', critic='bilinear', features=features,
             nlayers=args.nlayers,
             root_dir=f'./models/{args.dataset}')
if args.training:
    model.train_model(adj, features,
                      dataset_name=dataset_name,
                      alpha=args.alpha,
                      beta=args.beta)

embeds = model.get_embed_from_dataset(adj, features, dataset_name=dataset_name, alpha=args.alpha,
                                      beta=args.beta)
acc = eval_clf(embeds, labels, train_mask, test_mask)
print('acc:', acc)
