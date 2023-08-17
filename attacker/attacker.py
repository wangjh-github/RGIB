import torch as ch
from tqdm import tqdm
from utils import helpers
from . import attack_steps
import time, gc


def mi_loss(encoder, perturbed_adj, perturbed_features, nb_nodes, b_xent, batch_size=1, sparse=True, encoder_cpu=False,
             ori_adj=None, ori_features=None):
    mi_loss, _, _ = encoder(ori_features, ori_adj, perturbed_features, perturbed_adj, sparse)
    return mi_loss


class Attacker(ch.nn.Module):
    """
    Attacker class, used to make adversarial examples.

    This is primarily an internal class, you probably want to be looking at
    :class:`robustness.attacker.AttackerModel`, which is how models are actually
    served (AttackerModel uses this Attacker class).

    However, the :meth:`robustness.Attacker.forward` function below
    documents the arguments supported for adversarial attacks specifically.
    """

    def __init__(self, model, features, nb_nodes, batch_size=1, sparse=True,
                 dataset=None, attack_mode='A', show_attack=True, gpu=True):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        # self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        # self.sp_adj = sp_adj
        # self.sp_A = sp_A
        # self.sp_adj_ori = sp_adj_ori
        self.features = features
        self.nb_nodes = nb_nodes
        self.batch_size = batch_size
        self.sparse = sparse
        self.dataset = dataset
        self.attack_mode = attack_mode
        self.show_attack = show_attack
        self.gpu = gpu
        self.mi_loss = mi_loss


        if self.gpu and not attack_mode=='X':
            self.I = ch.eye(self.nb_nodes)
            self.I = self.I.cuda()

    def forward(self, adj_sys, A, target, eps, step_size, iterations, b_xent, features=None, model=None,
                eps_x=0.1, step_size_x=1e-5,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True, return_image=True, est_grad=None,
                make_adv=True, return_a=False):

        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = adj_sys
        # orig_input = orig_input.cuda()
        # A_norm = self.preprocess(A)

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1
        # if features is None or not features is None:
        #     features = features
        if self.model is None or not model is None:
            self.model = model

        # Main function for making adversarial examples
        def get_adv_examples(adj, A, return_a=False):
            # Initialize step class and attacker criterion
            step_class = attack_steps.L0Step
            step = step_class(orig_input=orig_input, A=A, eps=eps, step_size=step_size, nb_nodes=self.nb_nodes)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_delta_A = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = loss.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterate
            # A = A.cuda()
            # self.I = self.I.cuda()
            C = -self.I.to(A.device) - 2 * A + 1
            delta_A = ch.autograd.Variable(ch.zeros(adj.shape, dtype=ch.float32), requires_grad=True)
            if self.gpu:
                delta_A = delta_A.cuda()
            for _ in iterator:
                delta_A = delta_A.detach().requires_grad_(True)

                adj = self.preprocess(delta_A * C + A)
                # adj = ch.matmul(ch.matmul(ch.diag(1 / ch.sqrt(ch.sum(adj + self.I, 0))),
                #                           (A + delta_A * (1 - 2 * A - self.I)) + self.I),
                #                 ch.diag(1 / ch.sqrt(ch.sum((A + delta_A * (1 - 2 * A - self.I)) + self.I, 0))))

                loss = self.mi_loss(self.model, adj, features, self.nb_nodes, b_xent, self.batch_size,
                                    self.sparse, ori_adj=adj_sys, ori_features=features)
                # loss = mi_loss_neg(self.model, x, self.sp_adj_ori, features, self.nb_nodes, b_xent, self.batch_size, self.sparse)
                # loss = loss.cpu()
                if self.show_attack:
                    print("     Attack Loss: {}".format(loss.detach().cpu().numpy()))

                if step.use_grad:
                    if est_grad is None:
                        grad, = ch.autograd.grad(m * loss, [delta_A], retain_graph=True)
                    else:
                        f = lambda _x: m * self.mi_loss(self.model, _x, features, self.nb_nodes, b_xent,
                                                        self.batch_size, self.sparse, ori_adj=adj_sys,
                                                        ori_features=features)
                        grad = helpers.calc_est_grad(f, adj, target, *est_grad)
                else:
                    grad = None

                with ch.no_grad():
                    args = [loss, best_loss, delta_A, best_delta_A]
                    best_loss, best_delta_A = replace_best(*args) if use_best else (loss, delta_A)

                    delta_A = step.step(delta_A, grad)
                    delta_A = step.project(delta_A)
                    if self.gpu:
                        delta_A = delta_A.cuda()
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))
                    grad.cpu()
                    del grad
                    gc.collect()
                    ch.cuda.empty_cache()

            # Save computation (don't compute last loss) if not use_best
            if not use_best:
                ret = adj.clone().detach()
                return step.to_image(ret) if return_image else ret

            loss = self.mi_loss(self.model, self.preprocess(step.to_image(delta_A) + A), features,
                                self.nb_nodes, b_xent, self.batch_size, self.sparse, ori_adj=adj_sys,
                                ori_features=features)
            args = [loss, best_loss, delta_A, best_delta_A]
            best_loss, best_delta_A = replace_best(*args)

            if return_a:
                return step.to_image(best_delta_A, show=True) + A if return_image else best_delta_A
            else:
                return self.preprocess(step.to_image(best_delta_A, show=True) + A) if return_image else best_delta_A
            # return self.preprocess(step.to_image(best_A, A_ori)) if return_image else best_A

        # Main function for making adversarial examples
        def get_adv_x_examples(x):
            # Initialize step class and attacker criterion
            stepx_class = attack_steps.LinfStep
            stepx = stepx_class(orig_input=orig_input, A=A, eps=eps, eps_x=eps_x, step_size=step_size,
                                step_size_x=step_size_x, nb_nodes=self.nb_nodes)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = loss.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterate
            for _ in iterator:
                # print(_)

                x = x.clone().detach().requires_grad_(True)
                # with ch.autograd.profiler.record_function("mi_loss"):
                loss = self.mi_loss(self.model, adj_sys, x, self.nb_nodes, b_xent, self.batch_size, self.sparse,
                                    ori_adj=adj_sys, ori_features=features)
                # loss = mi_loss_neg(self.model, x, self.sp_adj_ori, features, self.nb_nodes, b_xent, self.batch_size, self.sparse)
                if self.show_attack:
                    print("     Attack Loss: {}".format(loss.detach().cpu().numpy()))

                if stepx.use_grad:
                    if est_grad is None:
                        # ch.cuda.empty_cache()
                        # with ch.autograd.profiler.record_function("getgrad"):
                        grad, = ch.autograd.grad(m * loss, [x], retain_graph=True)

                    else:
                        f = lambda _x: m * self.mi_loss(self.model, adj_sys, x, self.nb_nodes, b_xent, self.batch_size,
                                                        self.sparse, ori_adj=adj_sys, ori_features=features)
                        grad = helpers.calc_est_grad(f, adj_sys, target, *est_grad)
                else:
                    grad = None

                with ch.no_grad():
                    args = [loss, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (loss, x)

                    x = stepx.step(x, grad)
                    x = stepx.project(x, features)
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

            # Save computation (don't compute last loss) if not use_best
            if not use_best:
                ret = x.clone().detach()
                return ret

            loss = self.mi_loss(self.model, adj_sys, x, self.nb_nodes,
                                b_xent, self.batch_size, self.sparse, ori_adj=adj_sys, ori_features=features)
            args = [loss, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return best_x

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch

        if self.attack_mode == 'A':
            adv_ret = get_adv_examples(adj_sys, A, return_a)
            return adv_ret
        if self.attack_mode == 'X':
            adv_X_ret = get_adv_x_examples(features)
            return adv_X_ret
        elif self.attack_mode == 'both':
            adv_ret = get_adv_examples(adj_sys, A, return_a)
            adv_X_ret = get_adv_x_examples(features)
            return adv_ret, adv_X_ret

    def preprocess(self, adj):
        adj = adj + self.I
        D = ch.diag(1 / ch.sqrt(ch.sum(adj, 0)))
        adj_sys = ch.matmul(ch.matmul(D, adj), D)
        # adj_sys = ch.matmul(ch.matmul(ch.diag(1 / ch.sqrt(ch.sum(adj + self.I, 0))), adj + self.I), ch.diag(1 / ch.sqrt(ch.sum(adj + self.I, 0))))
        return adj_sys
