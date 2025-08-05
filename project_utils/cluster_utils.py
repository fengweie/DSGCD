from __future__ import division, print_function
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
from scipy.optimize import linear_sum_assignment as linear_assignment
import random
import os
import argparse

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from sklearn import metrics
import time
import torch
import torch.fft
import numpy as np
import faiss
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""
import torch
import torch
import torch.nn.functional as F
import random

import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn.functional as F
import random
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
# ImageNet code should change this value
IMAGE_SIZE = 32
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)
def extract_amplitude(imgs):  # imgs: [B, C, H, W]
    fft = torch.fft.fft2(imgs, dim=(-2, -1))
    amp = torch.sqrt(fft.real**2 + fft.imag**2)
    amp_flat = amp.view(imgs.size(0), -1)
    return amp_flat


def compute_density_scores(amp_ulb, amp_lb, k=5):
    amp_ulb_np = normalize(amp_ulb.cpu().numpy(), axis=1)
    amp_lb_np = normalize(amp_lb.cpu().numpy(), axis=1)
    index = faiss.IndexFlatIP(amp_lb_np.shape[1])
    index.add(amp_lb_np)
    D, _ = index.search(amp_ulb_np, k)
    return D.mean(1)  # 

def separate_known_unknown(density_scores):
    gmm = GaussianMixture(n_components=2, random_state=0).fit(density_scores.reshape(-1,1))
    probs = gmm.predict_proba(density_scores.reshape(-1,1))
    unknown_component = np.argmin(gmm.means_)  # 
    probs_unknown = probs[:, unknown_component]
    return probs_unknown

def split_ulb_known_unknown(images, mask_lab, threshold=0.5, k=5):
    mask_all = torch.cat([mask_lab for _ in range(2)], dim=0)
    images_lb = images[mask_all]     # 
    images_ulb = images[~mask_all]   # 

    amp_lb = extract_amplitude(images_lb)
    amp_ulb = extract_amplitude(images_ulb)

    scores = compute_density_scores(amp_ulb, amp_lb, k)
    probs_unknown = separate_known_unknown(scores)

    probs_unknown = torch.tensor(probs_unknown, device=images.device)
    mask_unknown = probs_unknown > threshold

    images_ulb_known = images_ulb[~mask_unknown]
    images_ulb_unknown = images_ulb[mask_unknown]

    return images_ulb_known, images_ulb_unknown, probs_unknown

def fft_decompose(x):  # x: [B, C, H, W]
    fft = torch.fft.fft2(x, dim=(-2, -1))
    amp = torch.abs(fft)
    phase = torch.angle(fft)
    return amp, phase

def fft_reconstruct(amp, phase):
    real = amp * torch.cos(phase)
    imag = amp * torch.sin(phase)
    fft_complex = torch.complex(real, imag)
    return torch.fft.ifft2(fft_complex, dim=(-2, -1)).real

def stylize_known_from_unknown(model, images_ulb_known, images_ulb_unknown):

    with torch.no_grad():
        _, out_known = model(images_ulb_known)  # shape: [B1, num_classes]
        _, out_unknown = model(images_ulb_unknown)  # shape: [B2, num_classes]


    pred_known = torch.argmax(out_known, dim=1)  # [B1]
    pred_unknown = torch.argmax(out_unknown, dim=1)  # [B2]

    stylized_list = []
    for i, c in enumerate(pred_known):
        candidates = (pred_unknown == c).nonzero(as_tuple=True)[0]
        if len(candidates) == 0:
            continue  # 

        j = random.choice(candidates.tolist())  # 

        x_k = images_ulb_known[i].unsqueeze(0)  # [1, C, H, W]
        x_u = images_ulb_unknown[j].unsqueeze(0)  # [1, C, H, W]

        amp_k, phase_k = fft_decompose(x_k)
        amp_u, _ = fft_decompose(x_u)

        stylized = fft_reconstruct(amp_u, phase_k)  # [1, C, H, W]
        stylized_list.append(stylized)

    if len(stylized_list) == 0:
        return None  # 
    else:
        return torch.cat(stylized_list, dim=0)  # [B*, C, H, W]


def fft_decompose(x):  # x: [B, C, H, W]
    fft = torch.fft.fft2(x, dim=(-2, -1))
    amp = torch.abs(fft)
    phase = torch.angle(fft)
    return amp, phase

def fft_reconstruct(amp, phase):
    real = amp * torch.cos(phase)
    imag = amp * torch.sin(phase)
    fft_complex = torch.complex(real, imag)
    return torch.fft.ifft2(fft_complex, dim=(-2, -1)).real

def amplitude_exchange_within_unknown(images_ulb_unknown):
    B = images_ulb_unknown.size(0)
    stylized_list = []

    amp_all, phase_all = fft_decompose(images_ulb_unknown)

    for i in range(B):

        idx_candidates = list(range(B))
        idx_candidates.remove(i)
        j = random.choice(idx_candidates)

        amp_j = amp_all[j].unsqueeze(0)     # [1, C, H, W]
        phase_i = phase_all[i].unsqueeze(0) # [1, C, H, W]

        stylized = fft_reconstruct(amp_j, phase_i)
        stylized_list.append(stylized)

    stylized_tensor = torch.cat(stylized_list, dim=0)  # [B, C, H, W]
    return stylized_tensor

def improved_info_nce_logits(features, features2, n_views=2, temperature=1.0, device='cuda'):
    """
    features:  (2B, d) tensor
    features2: (2B, d) tensor
    """
    assert features.shape == features2.shape, "features and features2 must have the same shape"
    
    B = features.size(0) // n_views  

    all_features = torch.cat([features, features2], dim=0)  #

    labels = torch.cat([torch.arange(B) for _ in range(n_views * 2)], dim=0)  #
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)  

    all_features = F.normalize(all_features, dim=1)

    similarity_matrix = torch.matmul(all_features, all_features.T)  # (4B, 4B)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)  

    logits = logits / temperature
    return logits, labels
def construct_gcd_loss(student_proj, student_out, teacher_out, class_labels, mask_lab, cluster_criterion, epoch, args):

    # clustering, sup
    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

    # clustering, unsup
    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
    cluster_loss += args.memax_weight * me_max_loss

    # represent learning, unsup
    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

    # representation learning, sup
    student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
    student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
    sup_con_labels = class_labels[mask_lab]
    sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)
    loss = 0
    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
    loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

    return loss, cls_loss, cluster_loss, sup_con_loss, contrastive_loss


def compute_known_unknown_unsup_losses(
    model,
    images_ulb_known,         
    images_ulb_unknown,       
    stylized_known,          
    stylized_unknown,         
    cluster_criterion,
    info_nce_logits,
    epoch,
    args
):
    # === Known Domain ===
    student_proj_k, student_out_k = model(images_ulb_known)
    with torch.no_grad():
        _, teacher_out_k = model(stylized_known)

    # clustering loss
    cluster_loss_k = cluster_criterion(student_out_k, teacher_out_k, epoch)
    avg_probs_k = (student_out_k / 0.1).softmax(dim=1).mean(dim=0)
    memax_k = -torch.sum(torch.log(avg_probs_k**(-avg_probs_k))) + math.log(len(avg_probs_k))
    cluster_loss_k += args.memax_weight * memax_k

    # contrastive loss
    contrastive_logits_k, contrastive_labels_k = info_nce_logits(features=F.normalize(student_proj_k, dim=-1))
    contrastive_loss_k = torch.nn.CrossEntropyLoss()(contrastive_logits_k, contrastive_labels_k)

    # === Unknown Domain ===
    student_proj_u, student_out_u = model(images_ulb_unknown)
    with torch.no_grad():
        _, teacher_out_u = model(stylized_unknown)

    # clustering loss
    cluster_loss_u = cluster_criterion(student_out_u, teacher_out_u, epoch)
    avg_probs_u = (student_out_u / 0.1).softmax(dim=1).mean(dim=0)
    memax_u = -torch.sum(torch.log(avg_probs_u**(-avg_probs_u))) + math.log(len(avg_probs_u))
    cluster_loss_u += args.memax_weight * memax_u

    # contrastive loss
    contrastive_logits_u, contrastive_labels_u = info_nce_logits(features=F.normalize(student_proj_u, dim=-1))
    contrastive_loss_u = torch.nn.CrossEntropyLoss()(contrastive_logits_u, contrastive_labels_u)

    return {
        'cluster_loss_known': cluster_loss_k,
        'contrastive_loss_known': contrastive_loss_k,
        'cluster_loss_unknown': cluster_loss_u,
        'contrastive_loss_unknown': contrastive_loss_u,
    }



def compute_total_stylized_loss(
    model,
    images_ulb_known,
    images_ulb_unknown,
    stylized_known,
    stylized_unknown,
    stylized_known_lb,
    mask_lab,
    class_labels,
    class_labels_ulb_known,
    cluster_criterion,
    info_nce_logits,
    SupConLoss,
    epoch,
    args
):
    # --- Stylized Known ---
    proj_known, out_known = model(stylized_known)
    with torch.no_grad():
        _, teacher_out_known = model(images_ulb_known)

    cluster_loss_known = cluster_criterion(out_known, teacher_out_known, epoch)
    avg_probs_known = (out_known / 0.1).softmax(dim=1).mean(dim=0)
    me_max_known = -torch.sum(torch.log(avg_probs_known ** (-avg_probs_known))) + math.log(len(avg_probs_known))
    cluster_loss_known += args.memax_weight * me_max_known

    proj_known = F.normalize(proj_known, dim=-1)
    proj_known = proj_known.unsqueeze(1).repeat(1, 2, 1)
    sup_con_loss_known = SupConLoss()(proj_known, labels=class_labels_ulb_known)

    # --- Stylized Unknown ---
    proj_unknown, out_unknown = model(stylized_unknown)
    with torch.no_grad():
        _, teacher_out_unknown = model(images_ulb_unknown)

    cluster_loss_unknown = cluster_criterion(out_unknown, teacher_out_unknown, epoch)
    avg_probs_unk = (out_unknown / 0.1).softmax(dim=1).mean(dim=0)
    me_max_unk = -torch.sum(torch.log(avg_probs_unk ** (-avg_probs_unk))) + math.log(len(avg_probs_unk))
    cluster_loss_unknown += args.memax_weight * me_max_unk

    contrastive_logits_unk, contrastive_labels_unk = info_nce_logits(features=F.normalize(proj_unknown, dim=-1))
    contrastive_loss_unknown = nn.CrossEntropyLoss()(contrastive_logits_unk, contrastive_labels_unk)

    # --- Stylized Known (Supervised) ---
    student_proj, student_out = model(stylized_known_lb)
    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

    student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
    student_proj = F.normalize(student_proj, dim=-1)
    sup_con_labels = class_labels[mask_lab]
    sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

    # --- Total Loss ---
    total_loss = 0
    total_loss += args.cluster_weight * (cluster_loss_known + cluster_loss_unknown)
    total_loss += args.unsup_contrastive_weight * contrastive_loss_unknown
    total_loss += args.sup_contrastive_weight * sup_con_loss_known
    total_loss += args.sup_weight * (cls_loss + sup_con_loss)

    return total_loss

def adaptive_image_sampler(images, student_proj_norm, student_out,class_labels, mask_all, num_samples=128):
    """

    Args:
        images (Tensor): [B, C, H, W] 
        student_proj_norm (Tensor): [B, D]
        student_out (Tensor): [B, C] 
        num_samples (int)

    Returns:
        sampled_images (Tensor): [num_samples, C, H, W]
        sampled_indices (Tensor): [num_samples]
        sampled_labels (Tensor): [num_samples] 
    """
    probs = F.softmax(student_out, dim=1)
    preds = torch.argmax(probs, dim=1)
    unique_classes = preds.unique()
    C_u = len(unique_classes)

    prototypes = torch.stack([
        student_proj_norm[preds == c].mean(dim=0) for c in unique_classes
    ], dim=0)  # [C_u, D]

    intra_var = torch.stack([
        ((student_proj_norm[preds == c] - prototypes[i])**2).sum(dim=1).mean()
        if (preds == c).sum() > 1 else torch.tensor(0.0, device=student_proj_norm.device)
        for i, c in enumerate(unique_classes)
    ], dim=0)  # [C_u]

    inter_sim = torch.stack([
        F.cosine_similarity(
            prototypes[i].unsqueeze(0), 
            torch.cat([prototypes[j].unsqueeze(0) for j in range(C_u) if j != i], dim=0)
        ).mean()
        for i in range(C_u)
    ], dim=0)  # [C_u]

    difficulty_scores = torch.exp(intra_var + inter_sim)
    sampling_probs = difficulty_scores / difficulty_scores.sum()

    sampled_class_indices = torch.multinomial(sampling_probs, num_samples=num_samples, replacement=True)
    sampled_indices = []
    for class_idx in sampled_class_indices:
        class_label = unique_classes[class_idx]
        class_mask = (preds == class_label).nonzero(as_tuple=True)[0]
        rand_index = class_mask[torch.randint(0, len(class_mask), (1,))].item()
        sampled_indices.append(rand_index)

    sampled_indices = torch.tensor(sampled_indices, device=images.device)
    sampled_images = images[sampled_indices]
    sampled_labels = class_labels[sampled_indices]
    mask_all = mask_all[sampled_indices]

    return sampled_images, sampled_indices, sampled_labels, mask_all
# -------------------------------
# Evaluation Criteria
# -------------------------------
def evaluate_clustering(y_true, y_pred):

    start = time.time()
    print('Computing metrics...')
    if len(set(y_pred)) < 1000:
        acc = cluster_acc(y_true.astype(int), y_pred.astype(int))
    else:
        acc = None

    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    print(f'Finished computing metrics {time.time() - start}...')

    return acc, nmi, ari, pur


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# -------------------------------
# Mixed Eval Function
# -------------------------------
def mixed_eval(targets, preds, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)

    # Labelled examples
    if mask.sum() == 0:  # All examples come from unlabelled classes

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int), preds.astype(int)), \
                                                         nmi_score(targets, preds), \
                                                         ari_score(targets, preds)

        print('Unlabelled Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'
              .format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

        # Also return ratio between labelled and unlabelled examples
        return (unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()

    else:

        labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask],
                                                               preds.astype(int)[mask]), \
                                                   nmi_score(targets[mask], preds[mask]), \
                                                   ari_score(targets[mask], preds[mask])

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                     preds.astype(int)[~mask]), \
                                                         nmi_score(targets[~mask], preds[~mask]), \
                                                         ari_score(targets[~mask], preds[~mask])

        # Also return ratio between labelled and unlabelled examples
        return (labelled_acc, labelled_nmi, labelled_ari), (
            unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


def PairEnum(x,mask=None):

    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))

    if mask is not None:

        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))

    return x1, x2


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))