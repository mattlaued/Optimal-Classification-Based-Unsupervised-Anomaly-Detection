# adapated from https://github.com/microsoft/EdgeML/blob/master/pytorch/edgeml_pytorch/trainer/drocclf_trainer.py#L185
import os
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from Utils.model_methods import PL_Model


def cal_precision_recall(positive_scores, far_neg_scores, close_neg_scores, fpr):
    """
    Computes the precision and recall for the given false positive rate.
    """
    # combine the far and close negative scores
    all_neg_scores = np.concatenate((far_neg_scores, close_neg_scores), axis=0)
    num_neg = all_neg_scores.shape[0]
    idx = int((1 - fpr) * num_neg)
    # sort scores in ascending order
    all_neg_scores.sort()
    thresh = all_neg_scores[idx]
    tp = np.sum(positive_scores > thresh)
    recall = tp / positive_scores.shape[0]
    fp = int(fpr * num_neg)
    precision = tp / (tp + fp)
    return precision, recall


def normalize_grads(grad):
    """
    Utility function to normalize the gradients.
    grad: (batch, -1)
    """
    # make sum equal to the size of second dim
    grad_norm = torch.sum(torch.abs(grad), dim=1)
    grad_norm = torch.unsqueeze(grad_norm, dim=1)
    grad_norm = grad_norm.repeat(1, grad.shape[1])
    grad = torch.nan_to_num(grad / grad_norm * grad.shape[1], nan=0.0)
    return grad


def compute_mahalanobis_distance(grad, diff, radius, device, gamma):
    """
    Compute the mahalanobis distance.
    grad: (batch,-1)
    diff: (batch,-1)
    """
    mhlnbs_dis = torch.sqrt(torch.sum(grad * diff ** 2, dim=1))
    # Categorize the batches based on mahalanobis distance
    # lamda = 1 : mahalanobis distance < radius
    # lamda = 2 : mahalanobis distance > gamma * radius
    lamda = torch.zeros((grad.shape[0], 1))
    lamda[mhlnbs_dis < radius] = 1
    lamda[mhlnbs_dis > (gamma * radius)] = 2
    return lamda, mhlnbs_dis


# The following are utitlity functions for checking the conditions in
# Proposition 1 in https://arxiv.org/abs/2002.12718


def check_left_part1_vec_batched(lam: torch.Tensor, grad: torch.Tensor, diff: torch.Tensor, radius: float, device: torch.device):
    """
    lam: (B, S)
    grad: (B, D)
    diff: (B, D)
    Returns: (B, S)
    """
    lam_exp = lam.unsqueeze(2)        # (B, S, 1)
    grad_exp = grad.unsqueeze(1)      # (B, 1, D)
    diff_exp = diff.unsqueeze(1)      # (B, 1, D)

    numerator = (diff_exp ** 2) * (lam_exp ** 2) * (grad_exp ** 2)
    denominator = (1 + lam_exp * grad_exp) ** 2 + 1e-10

    term = numerator / denominator
    return term.sum(dim=2)            # (B, S)



def check_left_part1_vec(lam, grad, diff, radius, device):
    # Part 1 condition value
    n1 = torch.outer(lam ** 2, diff ** 2 * grad ** 2)
    d1 = (1 + torch.outer(lam, grad)) ** 2 + 1e-10
    term = torch.div(n1, d1)
    term_sum = torch.sum(term, dim=1)
    return term_sum


def check_left_part1(lam, grad, diff, radius, device):
    # Part 1 condition value
    n1 = diff ** 2 * lam ** 2 * grad ** 2
    d1 = (1 + lam * grad) ** 2 + 1e-10
    term = n1 / d1
    term_sum = torch.sum(term)
    return term_sum


def check_left_part2_vec_batched(nu: torch.Tensor, grad: torch.Tensor, diff: torch.Tensor, radius: float, device: torch.device, gamma: float):
    """
    nu: (B, S)
    grad: (B, D)
    diff: (B, D)
    Returns: (B, S)
    """
    nu_exp = nu.unsqueeze(2)         # (B, S, 1)
    grad_exp = grad.unsqueeze(1)     # (B, 1, D)
    diff_exp = diff.unsqueeze(1)     # (B, 1, D)

    numerator = diff_exp ** 2 * grad_exp ** 2
    denominator = (nu_exp + grad_exp) ** 2 + 1e-10

    term = numerator / denominator
    return term.sum(dim=2)           # (B, S)


def check_left_part2_vec(nu, grad, diff, radius, device, gamma):
    # Part 2 condition value
    n1 = diff ** 2 * grad ** 2
    d1 = (nu + grad) ** 2 + 1e-10
    term = n1 / d1
    term_sum = torch.sum(term, dim=1)
    return term_sum


def check_left_part2(nu, grad, diff, radius, device, gamma):
    # Part 2 condition value
    n1 = diff ** 2 * grad ** 2
    d1 = (nu + grad) ** 2 + 1e-10
    term = n1 / d1
    term_sum = torch.sum(term)
    return term_sum


def check_right_part1_vec_batched(lam: torch.Tensor, grad: torch.Tensor, diff: torch.Tensor, radius: float, device: torch.device):
    """
    lam: (B, S)
    grad: (B, D)
    diff: (B, D)
    Returns: (B, S)
    """
    B, S = lam.shape
    _, D = grad.shape

    lam_exp = lam.unsqueeze(2)       # (B, S, 1)
    grad_exp = grad.unsqueeze(1)     # (B, 1, D)
    diff_exp = diff.unsqueeze(1)     # (B, 1, D)

    numerator = (diff_exp ** 2) * grad_exp                   # (B, S, D)
    denominator = (1 + lam_exp * grad_exp) ** 2 + 1e-10      # (B, S, D)

    term = numerator / denominator
    term_sum = term.sum(dim=2)        # (B, S)

    left_check = check_left_part1_vec_batched(lam, grad, diff, radius, device)
    return torch.where(term_sum > radius ** 2, left_check, torch.full_like(term_sum, float('inf')))



def check_right_part1_vec(lam, grad, diff, radius, device):
    # Check if 'such that' condition is true in proposition 1 part 1
    # m
    n1 = grad
    # n x m
    d1 = (1 + torch.outer(lam, grad)) ** 2 + 1e-10
    # d1 = (1 + lam * grad) ** 2 + 1e-10
    # diff is size m
    term = torch.div((diff ** 2 * n1).reshape(1, -1), d1)       # n x m
    term_sum = torch.sum(term, dim=1)                          # n
    return torch.where(term_sum > radius ** 2, check_left_part1_vec(lam, grad, diff, radius, device), np.inf)
    # if term_sum > radius ** 2:
    #     return check_left_part1(lam, grad, diff, radius, device)
    # else:
    #     return np.inf



def check_right_part1(lam, grad, diff, radius, device):
    # Check if 'such that' condition is true in proposition 1 part 1
    n1 = grad
    d1 = (1 + lam * grad) ** 2 + 1e-10
    term = diff ** 2 * n1 / d1
    term_sum = torch.sum(term)
    if term_sum > radius ** 2:
        return check_left_part1(lam, grad, diff, radius, device)
    else:
        return np.inf


def check_right_part2_vec_batched(nu: torch.Tensor, grad: torch.Tensor, diff: torch.Tensor, radius: float, device: torch.device, gamma: float):
    """
    nu: (B, S)
    grad: (B, D)
    diff: (B, D)
    Returns: (B, S)
    """
    B, S = nu.shape
    _, D = grad.shape

    nu_exp = nu.unsqueeze(2)           # (B, S, 1)
    grad_exp = grad.unsqueeze(1)       # (B, 1, D)
    diff_exp = diff.unsqueeze(1)       # (B, 1, D)

    numerator = diff_exp ** 2 * (nu_exp * grad_exp) ** 2
    denominator = (nu_exp + grad_exp) ** 2 + 1e-10

    term = numerator / denominator
    term_sum = term.sum(dim=2)         # (B, S)

    left_check = check_left_part2_vec_batched(nu, grad, diff, radius, device, gamma)
    return torch.where(term_sum < (gamma * radius) ** 2, left_check, torch.full_like(term_sum, float('inf')))


def check_right_part2_vec(nu, grad, diff, radius, device, gamma):
    # Check if 'such that' condition is true in proposition 1 part 2
    n1 = torch.outer(nu, grad) ** 2
    d1 = (nu + grad) ** 2 + 1e-10
    term = torch.div(diff ** 2 * n1, d1)
    term_sum = torch.sum(term, dim=1)
    return torch.where(term_sum < radius ** 2, check_left_part2_vec(nu, grad, diff, radius, device, gamma), np.inf)
    # if term_sum < (gamma * radius) ** 2:
    #     return check_left_part2_vec(nu, grad, diff, radius, device, gamma)
    # else:
    #     # return torch.tensor(float('inf'))
    #     return np.inf


def check_right_part2(nu, grad, diff, radius, device, gamma):
    # Check if 'such that' condition is true in proposition 1 part 2
    n1 = grad * nu ** 2
    d1 = (nu + grad) ** 2 + 1e-10
    term = diff ** 2 * n1 / d1
    term_sum = torch.sum(term)
    if term_sum < (gamma * radius) ** 2:
        return check_left_part2(nu, grad, diff, radius, device, gamma)
    else:
        # return torch.tensor(float('inf'))
        return np.inf


def range_lamda_lower(grad):
    # Gridsearch range for lamda
    lam, _ = torch.max(grad, dim=1)
    eps, _ = torch.min(grad, dim=1)
    lam = -1 / lam + eps * 0.0001
    return lam


def range_nu_upper(grad, mhlnbs_dis, radius, gamma):
    # Gridsearch range for nu
    alpha = (gamma * radius) / mhlnbs_dis
    max_sigma, _ = torch.max(grad, dim=1)
    nu = (alpha / (1 - alpha)) * max_sigma
    return nu


def optim_solver(grad, diff, radius, device, gamma=2.):
    """
    Solver for the optimization problem presented in Proposition 1 in
    https://arxiv.org/abs/2002.12718
    """
    lamda, mhlnbs_dis = compute_mahalanobis_distance(grad, diff, radius, device, gamma)
    lamda_lower_limit = range_lamda_lower(grad).detach().cpu().numpy()
    nu_upper_limit = range_nu_upper(grad, mhlnbs_dis, radius, gamma).detach().cpu().numpy()

    # num of values of lamda and nu samples in the allowed range
    num_rand_samples = 40
    final_lamda = torch.zeros((grad.shape[0], 1))
    vec = True

    # Solve optim for each example in the batch
    for idx in range(lamda.shape[0]):
        # Optim corresponding to mahalanobis dis < radius
        if lamda[idx] == 1:
            if vec:
                # min_left = np.ones(num_rand_samples) * np.inf
                # best_lam = np.zeros(num_rand_samples)
                l = lamda_lower_limit[idx]
                if l >= 0.:
                    val = np.zeros(num_rand_samples)
                    left_val = 0.
                else:
                    if l == -np.inf:
                        # if l is -inf, set to a big number to prevent overflow issues when sampling
                        l = -1e30
                    val = np.random.uniform(low=l, high=0, size=num_rand_samples)
                    left_val = check_right_part1_vec(val, grad[idx], diff[idx], radius, device)
                # indices = left_val < min_left
                # min_left[indices] = left_val[indices]
                # best_lam[indices] = val[indices]
                best_left = np.argmin(left_val)
                best_lam = val[best_left]
            else:
                min_left = np.inf
                best_lam = 0.
                for k in range(num_rand_samples):
                    l = lamda_lower_limit[idx]
                    if l >= 0.:
                        val = 0.
                        left_val = 0.
                    else:
                        if l == -np.inf:
                            # if l is -inf, set to a big number to prevent overflow issues when sampling
                            l = -1e30
                        val = np.random.uniform(low=l, high=0)
                        left_val = check_right_part1(val, grad[idx], diff[idx], radius, device)
                    if left_val < min_left:
                        min_left = left_val
                        best_lam = val

            final_lamda[idx] = best_lam

        # Optim corresponding to mahalanobis dis > gamma * radius
        elif lamda[idx] == 2:
            if vec:
                # min_left = np.ones(num_rand_samples) * np.inf
                # best_lam = np.zeros(num_rand_samples)
                val = np.random.uniform(low=0, high=nu_upper_limit[idx], size=num_rand_samples)
                left_val = check_right_part2_vec(val, grad[idx], diff[idx], radius, device, gamma)
                # indices = left_val < min_left
                # min_left[indices] = left_val[indices]
                # best_lam[indices] = val[indices]
                best_left = np.argmin(left_val)
                best_lam = val[best_left]
            else:
                min_left = np.inf
                best_lam = np.inf
                for k in range(num_rand_samples):
                    val = np.random.uniform(low=0, high=nu_upper_limit[idx])
                    left_val = check_right_part2(val, grad[idx], diff[idx], radius, device, gamma)
                    if left_val < min_left:
                        min_left = left_val
                        best_lam = val

            final_lamda[idx] = 1.0 / best_lam

        else:
            final_lamda[idx] = 0

    final_lamda = final_lamda.to(device)
    for j in range(diff.shape[0]):
        diff[j, :] = diff[j, :] / (1 + final_lamda[j] * grad[j, :])

    return diff


def optim_solver_vec(grad, diff, radius, device, gamma=2., num_rand_samples=40):
    """
    Solver for the optimization problem presented in Proposition 1 in
    https://arxiv.org/abs/2002.12718
    """
    lamda, mhlnbs_dis = compute_mahalanobis_distance(grad, diff, radius, device, gamma)
    lamda_lower_limit = range_lamda_lower(grad).detach().cpu().numpy()
    nu_upper_limit = range_nu_upper(grad, mhlnbs_dis, radius, gamma).detach().cpu().numpy()

    # num of values of lamda and nu samples in the allowed range

    # Initialize final lamda
    final_lamda = torch.zeros((grad.shape[0], 1), device=device)


    # ==== CASE 1: lamda == 1 ====
    mask_1 = (lamda == 1)
    idxs_1 = mask_1.nonzero(as_tuple=True)[0]

    if idxs_1.numel() > 0:
        grad_1 = grad[idxs_1]
        diff_1 = diff[idxs_1]
        lamda_lower_1 = lamda_lower_limit[idxs_1]

        lamda_samples = np.zeros((len(idxs_1), num_rand_samples))
        try:
            enum = enumerate(lamda_lower_1)
        except:
            enum = enumerate([lamda_lower_1])
        for i, l in enum:
            if l == -np.inf:
                l = -1e30
            if l < 0.:
                lamda_samples[i] = np.random.uniform(low=l, high=0, size=num_rand_samples)

        lamda_samples_tensor = torch.tensor(lamda_samples, device=device, dtype=torch.float32)

        vals = check_right_part1_vec_batched(lamda_samples_tensor, grad_1, diff_1, radius, device)
        best_idxs = torch.argmin(vals, dim=1)
        best_lam = lamda_samples_tensor[torch.arange(len(idxs_1)), best_idxs]

        final_lamda[idxs_1] = best_lam.unsqueeze(1)

    # ==== CASE 2: lamda == 2 ====
    mask_2 = (lamda == 2)
    idxs_2 = mask_2.nonzero(as_tuple=True)[0]

    if idxs_2.numel() > 0:
        grad_2 = grad[idxs_2]
        diff_2 = diff[idxs_2]
        nu_upper_2 = nu_upper_limit[idxs_2]

        nu_samples = np.random.uniform(low=0.0, high=1.0, size=(len(idxs_2), num_rand_samples))
        try:
            enum = enumerate(nu_upper_2)
        except:
            enum = enumerate([nu_upper_2])
        for i, max_val in enum:
            nu_samples[i] *= max_val

        nu_samples_tensor = torch.tensor(nu_samples, device=device, dtype=torch.float32)

        vals = check_right_part2_vec_batched(nu_samples_tensor, grad_2, diff_2, radius, device, gamma)
        best_idxs = torch.argmin(vals, dim=1)
        best_nu = nu_samples_tensor[torch.arange(len(idxs_2)), best_idxs]

        final_lamda[idxs_2] = (1.0 / best_nu).unsqueeze(1)

    # ==== CASE 3: lamda == 0 (no-op) ====
    # final_lamda already initialized to zero

    # ==== APPLY final lamda to perturbation ====
    diff = diff / (1 + final_lamda * grad)

    return diff.to(device)


def get_gradients(model, device, data, target):
    """
    Utility function to compute the gradients of the model on the
    given data.
    """
    total_train_pts = len(data)
    data = data.to(torch.float)
    target = target.to(torch.float)
    target = torch.squeeze(target)

    # Extract the logits for cross entropy loss
    data_copy = data
    data_copy = data_copy.detach().requires_grad_()
    # logits = model(data_copy)
    logits = model(data_copy)
    logits = torch.squeeze(logits, dim=1)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target)

    grad = torch.autograd.grad(ce_loss, data_copy)[0]

    return torch.abs(grad)


class DROCC_LF(PL_Model):
    def __init__(self, backbone, classifier, optimizer=torch.optim.Adam, optimizer_params={"lr": 1e-3},
                 lr_scheduler=None, lr_scheduler_params=dict(), attack=None,
                 positive_class=1, loss_fn=torch.nn.functional.binary_cross_entropy,
                 freeze_backbone=True, use_hidden_layer_of_backbone=True,
                 neg_labels=False, seed=42, device='cuda', lamda=1, radius=0.2, gamma=2.0,
                 only_ce_epochs=50,
                 ascent_step_size=0.001, ascent_num_steps=50):
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.bce = loss_fn

        self.only_ce_epochs = only_ce_epochs
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        super(DROCC_LF, self).__init__(backbone, classifier, optimizer=optimizer, optimizer_params=optimizer_params,
                 lr_scheduler=lr_scheduler, lr_scheduler_params=lr_scheduler_params, attack=attack,
                 positive_class=positive_class, loss_fn=None,
                 freeze_backbone=freeze_backbone, use_hidden_layer_of_backbone=use_hidden_layer_of_backbone,
                 neg_labels=neg_labels, seed=seed, device=device)

    def get_loss(self, x, y, only_ce_epochs=None, train=True):
        if only_ce_epochs is None:
            only_ce_epochs = self.only_ce_epochs
        # Extract the pred for cross entropy loss
        y_pred = self.forward(x)
        # y_pred = torch.squeeze(y_pred, dim=1)
        ce_loss = self.bce(y_pred, y)

        '''
        Adversarial Loss is calculated only for the positive data points (label==1).
        '''
        if self.current_epoch >= only_ce_epochs:
            # device = self.device
            # device = 'cpu'
            pos_data = x[y == 1]#.to(device)
            if len(pos_data) == 0:
                if not train:
                    return ce_loss
            target = torch.ones(pos_data.shape[0]).to(self.device)
            gradients = self.get_gradients(pos_data, target)
            # AdvLoss
            adv_loss = self.one_class_adv_loss(pos_data, gradients)

            loss = ce_loss + adv_loss * self.lamda
        else:
            # If only CE based training has to be done
            loss = ce_loss
        return loss


    def training_step(self, batch, batch_idx, adv_training=False, model=None):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # add it here in case we need to do adversarial training later on
        if adv_training and model is not None:
            x = self.attack(model, x, y, epsilon=0.03, alpha=0.01, num_iter=40, random_start=True,
                            device=self.device)
        # y_pred = self.forward(x)
        loss = self.get_loss(x, y)
        # if self.neg_labels:
        #     y = 2 * y - 1
        #     y_pred = 2 * y_pred - 1
        # loss = self.loss_fn(y_pred, y)
        #         loss = self.get_loss(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        # if self.positive_class == 0:
        #     y = 1. - y
        #     y_pred = 1. - y_pred
        # self.log_metrics(y_pred, y, prefix="train")
        return loss

    @torch.inference_mode(False)
    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        # y_pred = self.forward(x)
        loss = self.get_loss(x, y, train=False)
        #         loss = self.get_loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        # if self.positive_class == 0:
        #     y = 1. - y
        #     y_pred = 1. - y_pred
        # self.log_metrics(y_pred, y, prefix="val")
        return loss

    def get_gradients(self, data, target):
        """
        Utility function to compute the gradients of the model on the
        given data.
        """
        # total_train_pts = len(data)
        data = data.to(torch.float)
        target = target.to(torch.float)
        target = torch.squeeze(target)

        # Extract the pred for cross entropy loss
        data_copy = data
        data_copy = data_copy.detach().requires_grad_()
        # logits = model(data_copy)
        y_pred = self.forward(data_copy)
        # y_pred = torch.squeeze(y_pred, dim=1)
        ce_loss = self.bce(y_pred, target)

        grad = torch.autograd.grad(ce_loss, data_copy)[0]

        return torch.abs(grad)

    def one_class_adv_loss(self, x_train_data, gradients):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r) with mahalanobis distance as a distance measure),
            by solving the optimization problem
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0

        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        gradients: gradients of the model for the given data.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():
                # try:
                new_targets = torch.zeros(batch_size).to(torch.float).to(self.device)
                # except:
                #     new_targets = torch.zeros(batch_size, 1)
                #     new_targets = torch.squeeze(new_targets)
                #     new_targets = new_targets.to(torch.float)
                #     new_targets = new_targets.to(self.device)

                y_pred = self.forward(x_adv_sampled)
                # y_pred = torch.squeeze(y_pred, dim=1)
                new_loss = self.bce(y_pred, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                # nan to 0 to account for where grad is 0
                #   (and so, grad_norm is also 0, producing nan due to divide by 0 error)
                grad_normalized = torch.nan_to_num(grad / grad_norm, nan=0.0)
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 5 == 0:
                # Project the normal points to the set N_i(r) based on mahalanobis distance
                h = x_adv_sampled - x_train_data
                h_flat = torch.reshape(h, (h.shape[0], -1))
                gradients_flat = torch.reshape(gradients, (gradients.shape[0], -1))
                # Normalize the gradients
                gradients_normalized = normalize_grads(gradients_flat)
                # Solve the non-convex 1D optimization
                h_flat = optim_solver_vec(gradients_normalized, h_flat, self.radius, self.device, self.gamma)
                h = torch.reshape(h_flat, h.shape)
                x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere
                del h, h_flat, gradients_flat, gradients_normalized

        adv_pred = self.forward(x_adv_sampled)
        # adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = self.bce(adv_pred, (new_targets * 0))

        return adv_loss


# trainer class for DROCC
class DROCCLFTrainer:
    """
    Trainer class that implements the DROCC-LF algorithm proposed for
    one-class classification with limited negative data presented in
    https://arxiv.org/abs/2002.12718
    """

    def __init__(self, model, optimizer, lamda, radius, gamma, device):
        """Initialize the DROCC-LF Trainer class

        Parameters
        ----------
        model: Torch neural network object
        optimizer: Total number of epochs for training.
        lamda: Weight given to the adversarial loss
        radius: Radius of hypersphere to sample points from.
        gamma: Parameter to vary projection.
        device: torch.device object for device to use.
        """
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.device = device

    def train(self, train_loader, val_loader, closeneg_val_loader, learning_rate, lr_scheduler, total_epochs,
              only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50):
        """Trains the model on the given training dataset with periodic
        evaluation on the validation dataset.

        Parameters
        ----------
        train_loader: Dataloader object for the training dataset.
        val_loader: Dataloader object for the validation dataset with far negatives.
        closeneg_val_loader: Dataloader object for the validation dataset with close negatives.
        learning_rate: Initial learning rate for training.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        ascent_step_size: Step size for gradient ascent for adversarial
                          generation of negative points.
        ascent_num_steps: Number of gradient ascent steps for adversarial
                          generation of negative points.
        """
        best_recall_fpr03 = -np.inf
        best_precision_fpr03 = -np.inf
        best_recall_fpr05 = -np.inf
        best_precision_fpr05 = -np.inf
        best_model = None
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        for epoch in range(total_epochs):
            # Make the weights trainable
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)

            # Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  # AdvLoss
            epoch_ce_loss = 0  # Cross entropy Loss

            batch_idx = -1
            for data, target, _ in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)

                self.optimizer.zero_grad()

                # Extract the logits for cross entropy loss
                logits = self.model(data)
                logits = torch.squeeze(logits, dim=1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                # Add to the epoch variable for printing average CE Loss
                epoch_ce_loss += ce_loss

                '''
                Adversarial Loss is calculated only for the positive data points (label==1).
                '''
                if epoch >= only_ce_epochs:
                    data = data[target == 1]
                    target = torch.ones(data.shape[0]).to(self.device)
                    gradients = get_gradients(self.model, self.device, data, target)
                    # AdvLoss
                    adv_loss = self.one_class_adv_loss(data, gradients)
                    epoch_adv_loss += adv_loss

                    loss = ce_loss + adv_loss * self.lamda
                else:
                    # If only CE based training has to be done
                    loss = ce_loss

                # Backprop
                loss.backward()
                self.optimizer.step()

            epoch_ce_loss = epoch_ce_loss / (batch_idx + 1)  # Average CE Loss
            epoch_adv_loss = epoch_adv_loss / (batch_idx + 1)  # Average AdvLoss

            # normal val loader has the positive data and the far negative data
            auc, pos_scores, far_neg_scores = self.test(val_loader, get_auc=True)
            _, _, close_neg_scores = self.test(closeneg_val_loader, get_auc=False)

            precision_fpr03, recall_fpr03 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.03)
            precision_fpr05, recall_fpr05 = cal_precision_recall(pos_scores, far_neg_scores, close_neg_scores, 0.05)
            if recall_fpr03 > best_recall_fpr03:
                best_recall_fpr03 = recall_fpr03
                best_precision_fpr03 = precision_fpr03
                best_recall_fpr05 = recall_fpr05
                best_precision_fpr05 = precision_fpr05
                best_model = copy.deepcopy(self.model)
            print('Epoch: {}, CE Loss: {}, AdvLoss: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item()))
            print('Precision @ FPR 3% : {}, Recall @ FPR 3%: {}'.format(
                precision_fpr03, recall_fpr03))
            print('Precision @ FPR 5% : {}, Recall @ FPR 5%: {}'.format(
                precision_fpr05, recall_fpr05))
        self.model = copy.deepcopy(best_model)
        print('\nBest test Precision @ FPR 3% : {}, Recall @ FPR 3%: {}'.format(
            best_precision_fpr03, best_recall_fpr03
        ))
        print('\nBest test Precision @ FPR 5% : {}, Recall @ FPR 5%: {}'.format(
            best_precision_fpr05, best_recall_fpr05
        ))

    def test(self, test_loader, get_auc=True):
        """Evaluate the model on the given test dataset.

        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        """
        label_score = []
        batch_idx = -1
        for data, target, _ in test_loader:
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            logits = torch.squeeze(logits, dim=1)
            sigmoid_logits = torch.sigmoid(logits)
            scores = sigmoid_logits
            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                    scores.cpu().data.numpy().tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        auc = -1
        if get_auc:
            auc = roc_auc_score(labels, scores)
        return auc, pos_scores, neg_scores

    def one_class_adv_loss(self, x_train_data, gradients):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r) with mahalanobis distance as a distance measure),
            by solving the optimization problem
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0

        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        gradients: gradients of the model for the given data.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():

                new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.model(x_adv_sampled)
                logits = torch.squeeze(logits, dim=1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim=tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1] * (grad.dim() - 1))
                grad_normalized = grad / grad_norm
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 5 == 0:
                # Project the normal points to the set N_i(r) based on mahalanobis distance
                h = x_adv_sampled - x_train_data
                h_flat = torch.reshape(h, (h.shape[0], -1))
                gradients_flat = torch.reshape(gradients, (gradients.shape[0], -1))
                # Normalize the gradients
                gradients_normalized = normalize_grads(gradients_flat)
                # Solve the non-convex 1D optimization
                h_flat = optim_solver(gradients_normalized, h_flat, self.radius, self.device, self.gamma)
                h = torch.reshape(h_flat, h.shape)
                x_adv_sampled = x_train_data + h  # These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))

        return adv_loss

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))