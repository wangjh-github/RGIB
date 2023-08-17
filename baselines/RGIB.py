import time

import torch
import torch.nn as nn

import scipy.sparse as sp
import numpy as np
import os
from attacker.attacker import Attacker
from utils import process
from .Eval import LogReg
from layers import GCN, AvgReadout, Discriminator
import pickle as pkl
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, from_scipy_sparse_matrix, to_scipy_sparse_matrix
from yaml import SafeLoader
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


class Encoder(torch.nn.Module):
    def __init__(self, in_ft, out_ft, act, n_layers=1, bias=True):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(GCN(in_ft, out_ft, act, bias=bias))
        for i in range(n_layers - 1):
            self.layers.append(GCN(out_ft, out_ft, act, bias=bias))

        self.porj = nn.Sequential(nn.Linear(out_ft, out_ft),
                                  nn.PReLU(),
                                  nn.Linear(out_ft, out_ft))

    def forward(self, x, adj, sparse):
        h = x
        for layer in self.layers:
            h = layer(h, adj, sparse)
        return h, h.squeeze(), None

    def summary(self, x, adj, sparse, with_fc=True):
        h = x
        for layer in self.layers:
            h = layer(h, adj, sparse, woact=True)
        if with_fc:
            h = self.porj(h)
        return h

    def embed(self, x, adj, sparse):
        h, mu, logvar = self(x, adj, sparse)
        return mu.squeeze().detach()


def drop_edge(adj, p):
    edge_index = adj._indices()
    edge_index, _ = dropout_adj(edge_index, p=p)
    return torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(edge_index.device), size=adj.shape)


class RGIB(torch.nn.Module):
    def __init__(self, n_in, n_h, activation, critic, features, nlayers=1, root_dir='.',
                  alpha=0.9, beta=0.1):
        super(RGIB, self).__init__()
        self.encoder = Encoder(n_in, n_h, activation, n_layers=nlayers)
        self.disc1 = Discriminator(n_h, critic=critic)
        self.disc2 = Discriminator(n_h, critic=critic)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        self.atm = Attacker(self.forward, None, features.shape[0], attack_mode="X",
                            show_attack=False, gpu=torch.cuda.is_available())
        self.model_path = f'{root_dir}/RGIB'

        self.n_in = n_in
        self.n_h = n_h

    def forward(self, x, adj, x_hat, adj_hat, sparse, samp_bias1=None, samp_bias2=None):
        assert len(x.shape) == 3, 'features has worng type'
        idx = np.random.permutation(x.shape[1])
        permutated_x = x_hat[:, idx, :]

        embbed, mu, logvar = self.encoder(x, adj, sparse)
        embbed_permutated, _, _ = self.encoder(permutated_x.detach(), adj_hat.detach(), sparse)
        embeded_polluted, mu_polluted, logvar_polluted = self.encoder(x_hat, adj_hat, sparse)
        summary = self.sigm(self.read(mu.unsqueeze(0), None))

        summary_subgraph = self.encoder.summary(x, adj, sparse)
        logits = self.disc1(summary_subgraph, embeded_polluted, embbed_permutated, samp_bias1, samp_bias2)

        lbl_1 = torch.ones(x.shape[0], x.shape[1])
        lbl_2 = torch.zeros(x.shape[0], x.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1).to(logits.device)
        b_xent = nn.BCEWithLogitsLoss()
        mi_loss = b_xent(logits, lbl)

        permutated_x = x[:, idx, :]
        embbed_permutated, _, _ = self.encoder(permutated_x.detach(), adj.detach(), sparse)
        logits_benign = self.disc2(summary, embbed, embbed_permutated, samp_bias1, samp_bias2)
        mi_loss_benign = b_xent(logits_benign, lbl)

        kl_loss = torch.sum((mu_polluted - mu) ** 2, 1).mean()  # The std is fixed

        return mi_loss, mi_loss_benign, kl_loss

    def get_embed_from_dataset(self, adj, features, sparse=True, alpha=0.1, beta=1.0, **kwargs):
        features = torch.unsqueeze(features, 0)
        adj = process.normalize_adj(adj)
        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = torch.FloatTensor(adj[np.newaxis])

        if torch.cuda.is_available():
            print('Using CUDA')
            self.cuda()
            features = features.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
        sp_adj = sp_adj.to_dense()
        self.load_state_dict(torch.load(f'{self.model_path}/model.pkl'))
        self.cuda()
        embeds = self.encoder.embed(features, sp_adj if sparse else adj, sparse)
        return embeds.squeeze().detach()

    def train_model(self, adj, features, sparse=True,
                    alpha=0.3, beta=1.0, dataset_name=None,
                    **kwargs):
        config = yaml.load(open('./configs/RGIB/config.yaml'), Loader=SafeLoader)[dataset_name]

        lr = 0.001
        l2_coef = 0.0

        step_size_init = 20
        attack_iters = 20
        stepsize_x = config['stepsize_x']
        nb_epochs = 200
        attack_rate = 0.05
        nb_edges = int(adj.sum() / 2)
        n_flips = int(nb_edges * attack_rate)

        A = adj.copy()
        adj = process.normalize_adj(adj)
        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_A = process.sparse_mx_to_torch_sparse_tensor(A)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()

        features = torch.unsqueeze(features, 0)
        features_ori = features.clone()
        sp_adj_ori = sp_adj.clone()

        sp_adj = sp_adj.cuda()
        sp_A = sp_A.cuda()
        features_ori = features_ori.cuda()
        sp_adj_ori = sp_adj_ori.cuda()

        optimiser = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.disc1.parameters()},
                                      {'params': self.disc2.parameters()}], lr=lr,
                                     weight_decay=l2_coef)
        if torch.cuda.is_available():
            self.encoder.cuda()
            self.disc1.cuda()
            self.disc2.cuda()
            self.atm.cuda()

        b_xent = nn.BCEWithLogitsLoss()

        for epoch in range(nb_epochs):
            self.train()
            optimiser.zero_grad()

            make_adv = (epoch % 1 == 0)
            if make_adv:
                step_size = step_size_init
                step_size_x = stepsize_x

                adv = self.atm(sp_adj.cuda(), sp_A.cuda(), None, n_flips, features=features_ori.cuda(),
                               model=self.forward, b_xent=b_xent, step_size=step_size,
                               eps_x=config['eps_x'], step_size_x=step_size_x,
                               iterations=attack_iters, should_normalize=True, random_restarts=False, make_adv=True)



                features = adv.detach()
                sp_adj = drop_edge(sp_A, config['p'])
                sp_adj = self.preprocess(sp_adj)


            mi_loss_adv, mi_loss_benign, kl_loss = self.forward(features_ori.cuda(), sp_adj_ori.cuda(), features.cuda(),
                                                                sp_adj.cuda(), sparse)

            loss = alpha * mi_loss_adv + beta * kl_loss + (1 - alpha) * mi_loss_benign
            print(
                f"Epoch:{epoch}, benign mi loss:{mi_loss_benign.item()}, adv mi loss:{mi_loss_adv.item()}, kl_loss:{kl_loss.item()}, Loss:{loss.item()}")

            loss.backward()
            optimiser.step()
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.state_dict(), f'{self.model_path}/model.pkl')

    def preprocess(self, adj):
        adj = torch.eye(adj.shape[0]).to(adj.device) + adj
        D = torch.diag(1 / torch.sqrt(torch.sum(adj, 0)))
        adj_sys = torch.matmul(torch.matmul(D, adj), D)
        return adj_sys
