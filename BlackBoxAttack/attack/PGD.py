import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
from . import attack_steps


def preprocess_features_tensor(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = features.sum(1)
    r_inv = torch.pow(rowsum, -1).squeeze()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv).to(features.device)
    features = torch.matmul(r_mat_inv, features)
    return features


class FPGDAttack(BaseAttack):
    """PGD attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import PGDAttack
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', attack_structure=False,
                 attack_features=True, device='cpu'):

        super(FPGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        # self.modified_adj = None
        self.modified_features = None

        # if attack_structure:
        #     assert nnodes is not None, 'Please give nnodes='
        #     self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
        #     self.adj_changes.data.fill_(0)

        # if attack_features:
        #     assert True, 'Topology Attack does not support attack feature'

        # self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_train, epochs=200, step_size_x=1e-3, eps_x=0.1,
               show_attack=False, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """

        victim_model = self.surrogate
        self.nb_nodes = ori_features.shape[0]

        # self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        ori_features = preprocess_features_tensor(ori_features)
        self.features = ori_features.clone()

        adj_norm = utils.normalize_adj_tensor(ori_adj)
        victim_model.eval()
        self.show_attack = show_attack

        def get_adv_x_examples(x):
            # Initialize step class and attacker criterion
            stepx_class = attack_steps.LinfStep
            stepx = stepx_class(orig_input=None, A=None, eps=0, eps_x=eps_x, step_size=0,
                                step_size_x=step_size_x, nb_nodes=self.nb_nodes)

            iterator = range(epochs)
            iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            m = 1
            # A function that updates the best loss and best input

            # PGD iterate
            for _ in iterator:
                x = x.clone().detach().to(self.device).requires_grad_(True)
                x = preprocess_features_tensor(x)

                output = victim_model(x, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                # loss = mi_loss(self.model, adj_sys, x, self.nb_nodes, b_xent, self.batch_size, self.sparse)
                # loss = mi_loss_neg(self.model, x, self.sp_adj_ori, self.features, self.nb_nodes, b_xent, self.batch_size, self.sparse)
                if self.show_attack:
                    print("     Attack Loss: {}".format(loss.detach().cpu().numpy()))

                if stepx.use_grad:
                    # ch.cuda.empty_cache()
                    grad, = torch.autograd.grad(m * loss, [x], retain_graph=True)

                else:
                    grad = None

                with torch.no_grad():
                    args = [loss, best_loss, x, best_x]
                    best_loss, best_x = (loss, x)

                    x = stepx.step(x, grad)
                    x = stepx.project(x, self.features)
                    iterator.set_description("Current loss: {l}".format(l=loss))
                    # print(torch.norm(x - ori_features, 2, dim=1))

            # Save computation (don't compute last loss) if not use_best
            ret = x.clone().detach()
            return ret

        self.modified_features = get_adv_x_examples(self.features)
        # self.modified_features = preprocess_features_tensor(self.modified_features)
        # print(torch.norm(self.modified_features - ori_features, 2, dim=1))


    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.cross_entropy(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                     output[np.arange(len(output)), best_second_class]
            k = 0.0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss
