from __future__ import print_function
from collections import OrderedDict
from functools import partial
import numpy as np
import math

from re import A

from timm.models.layers import Mlp, DropPath

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
from einops import rearrange

from methods.ours.loss import *
from models import vision_transformer as vits

from project_utils.general_utils import finetune_params

from loss import info_nce_logits, SupConLoss


class CrossEntropyMixup(nn.Module):

    def __init__(self, num_classes):
        super(CrossEntropyMixup, self).__init__()
        self.num_classes = num_classes
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, s_lambda=None):

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        targets = targets.to(inputs.device)
        s_lambda = s_lambda.unsqueeze(1)
        targets = s_lambda * targets + (1 - s_lambda) / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        return loss.mean()
    

def cosine_distance(source_hidden_features, target_hidden_features):
    "similarity between different features"
    n_s = source_hidden_features.shape[0]
    n_t = target_hidden_features.shape[0]
    
    temp_matrix = torch.mm(source_hidden_features, target_hidden_features.t())

    for i in range(n_s):
        vec = source_hidden_features[i]
        temp_matrix[i] /= torch.norm(vec, p=2)
    for j in range(n_t):
        vec = target_hidden_features[j]
        temp_matrix[:, j] /= torch.norm(vec, p=2)
    return temp_matrix

def convert_to_onehot(s_label, class_num):
    s_sca_label = s_label.cpu().data.numpy()
    return np.eye(class_num)[s_sca_label]

def mixup_soft_ce(pred, targets, weight, lam):
    """ mixed categorical cross-entropy loss
    """
    loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, targets)
    loss = torch.sum(lam*weight*loss) / (torch.sum(weight*lam).item())
    loss = loss * torch.sum(lam)
    return loss

def mixup_sup_dis(preds, s_label, lam):
    """ mixup_distance_in_feature_space_for_intermediate_source
    """
    label = torch.mm(s_label, s_label.t())
    mixup_loss = -torch.sum(label * F.log_softmax(preds), dim=1)
    mixup_loss = torch.sum (torch.mul(mixup_loss, lam))
    return mixup_loss

def mixup_unsup_dis(preds, lam):
    """ mixup_distance_in_feature_space_for_intermediate_target
    """
    label = torch.eye(preds.shape[0]).cuda()
    mixup_loss = -torch.sum(label* F.log_softmax(preds), dim=1)
    mixup_loss = torch.sum(torch.mul(mixup_loss, lam))
    return mixup_loss

def mix_token(s_token, t_token, s_lambda, t_lambda):
    # print(s_token.shape, s_lambda.shape, t_token.shape)
    s_token = torch.einsum('BNC,BN -> BNC', s_token, s_lambda)
    t_token = torch.einsum('BNC,BN -> BNC', t_token, t_lambda)
    m_tokens = s_token+t_token
    return m_tokens

def mix_lambda_atten(s_scores, t_scores, s_lambda, num_patch):
    t_lambda = 1-s_lambda
    if s_scores is None or t_scores is None:
        s_lambda = torch.sum(s_lambda, dim=1) / num_patch # important for /self.num_patch
        t_lambda = torch.sum(t_lambda, dim=1) / num_patch
        s_lambda = s_lambda / (s_lambda+t_lambda)        
    else:
        s_lambda = torch.sum(torch.mul(s_scores, s_lambda), dim=1) / num_patch # important for /self.num_patch
        t_lambda = torch.sum(torch.mul(t_scores, t_lambda), dim=1) / num_patch
        s_lambda = s_lambda / (s_lambda+t_lambda)
    return s_lambda


def mix_lambda (s_lambda,t_lambda):
    return torch.sum(s_lambda,dim=1) / (torch.sum(s_lambda,dim=1) + torch.sum(t_lambda,dim=1))


def softplus(x):
    return  torch.log(1+torch.exp(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        save = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, save 


class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        t, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(t)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn     


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits
    

class PMTrans(nn.Module):
    '''
    Modified from the original PMTrans (https://arxiv.org/abs/2303.13434) 
    '''
    def __init__(self, pretrain_path, args):
        super(PMTrans, self).__init__()
        
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

        state_dict = torch.load(pretrain_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3

        for m in self.backbone.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in self.backbone.named_parameters():
            if args.model == 'dino':
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True

            elif args.model == 'clip':
                if 'transformer.resblocks' in name:
                    block_num = int(name.split('.')[2])
                    if block_num >= args.grad_from_block:
                        m.requires_grad = True
        self.sem_head = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
        
#         self.backbone = vits.__dict__['vit_base']()
#         state_dict = torch.load(pretrain_path, map_location='cpu')
#         self.backbone.load_state_dict(state_dict)
#         finetune_params(self.backbone, args) # HOW MUCH OF BASE MODEL TO FINETUNE

#         self.sem_head = DINOHead(in_dim=768, out_dim=args.num_ctgs, nlayers=3)
#         print("use simgcd")
    def forward(self, mixture):
        device = mixture.device
        mixed_tokens = self.backbone(mixture)
#         print(mixed_tokens.shape)
        student_proj, student_out = self.sem_head(mixed_tokens)      
        return student_proj, student_out
      