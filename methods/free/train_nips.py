import numpy as np
import matplotlib.pyplot as plt
import cv2

import argparse
import json
import faiss
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import torch.distributed as dist
from easydict import EasyDict as edict

from sklearn.mixture import GaussianMixture
from methods.simgcd.models.swin_pm import PMTrans, DINOHead

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import str2bool, AverageMeter, get_params_groups
from project_utils.cluster_utils import *
from loss import info_nce_logits, SupConLoss, ContrastiveLearningViewGenerator, DistillLoss, Distangleloss, MCC_DALN

from config import distortions, severity, ovr_envs, dino_pretrain_path

from sklearn.mixture import GaussianMixture
import random
import torchvision.transforms as T
augmentations = [autocontrast, equalize, posterize, rotate, solarize, color, contrast, brightness, sharpness]
augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]



def train(model, train_loader, optimizer, exp_lr_scheduler, cluster_criterion, epoch, args):
    class_weight_src = torch.ones(args.num_labeled_classes, ).cuda()

    loss_record = AverageMeter()

    model.train()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        images, class_labels, uq_idxs, mask_lab, domain_lab = batch
        mask_lab = mask_lab[:, 0]

        class_labels, mask_lab, uq_idxs = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool(), uq_idxs.cuda(non_blocking=True)
      
        images = torch.cat(images, dim=0).cuda(non_blocking=True)

        img_idx = torch.cat([uq_idxs[~mask_lab] for _ in range(2)], dim=0).cuda(non_blocking=True)
        label_source = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
        mask_all = torch.cat([mask_lab for _ in range(2)], dim=0)

        with torch.amp.autocast(device_type='cuda', enabled=(args.fp16_scaler is not None)):
            
            student_proj, student_out = model(images)
            images, sampled_indices, class_labels,mask_all = adaptive_image_sampler(images, student_proj_norm, student_out, class_labels, mask_all)

            images_ulb_known, images_ulb_unknown, probs_unknown = split_ulb_known_unknown(images, mask_all)
            stylized_known_lb = stylize_known_from_unknown(model, images[mask_all], images_ulb_unknown)
            stylized_known = stylize_known_from_unknown(model, images_ulb_known, images_ulb_unknown)

            stylized_unknown = amplitude_exchange_within_unknown(images_ulb_unknown)
            
            student_proj, student_out = model(images)
            
            student_proj_norm = torch.nn.functional.normalize(student_proj, dim=-1)
            loss_sim, cls_loss, cluster_loss, sup_con_loss, contrastive_loss = construct_gcd_loss(student_proj, student_out, student_out.detach(), class_labels, mask_lab, cluster_criterion, epoch, args)
            
            loss_freq = compute_total_stylized_loss(
                model=model,
                images_ulb_known=images_ulb_known,
                images_ulb_unknown=images_ulb_unknown,
                stylized_known=stylized_known,
                stylized_unknown=stylized_unknown,
                stylized_known_lb=stylized_known_lb,
                mask_lab=mask_lab,
                class_labels=class_labels,
                class_labels_ulb_known=class_labels_ulb_known,
                cluster_criterion=cluster_criterion,
                info_nce_logits=info_nce_logits,
                SupConLoss=SupConLoss,
                epoch=epoch,
                args=args
            )
            
            loss = loss_freq+loss_sim

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{}][{}/{}]\t loss {:.5f}\t'
                        .format(epoch, batch_idx, len(train_loader), loss.item()))                             
        # Train acc
        loss_record.update(loss.item(), class_labels.size(0))
        optimizer.zero_grad()
        if args.fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            args.fp16_scaler.scale(loss).backward()
            args.fp16_scaler.step(optimizer)
            args.fp16_scaler.update()

    # Step schedule
    exp_lr_scheduler.step()


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_partial_dataset', type=str2bool, default=False)
    parser.add_argument('--use_uda_loss', type=str2bool, default=False)

    parser.add_argument('--dataset_name', type=str, default='domainnet', help='options: ')
    parser.add_argument('--src_env', type=str)
    parser.add_argument('--tgt_env', type=str)
    parser.add_argument('--aux_env', type=str, default=None)
    parser.add_argument('--model', type=str, default='dino')
    parser.add_argument('--task_type', type=str, default='A_L+A_U->B')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--pre_splits', type=str2bool, default=False)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--lamb', type=float, default=0.1, help='The balance factor.')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--print_freq', default=500, type=int)
    
    parser.add_argument('--alpha', type=float, default=0.8, help='hyper-parameters alpha')
    parser.add_argument('--beta', type=float, default=3, help='hyper-parameters beta')

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--cluster_weight', type=float, default=1.0)
    parser.add_argument('--unsup_contrastive_weight', type=float, default=1.0)
    parser.add_argument('--sup_contrastive_weight', type=float, default=1.0)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--memax_weight_dom', type=float, default=0.1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=True)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda')
    args = get_class_splits(args)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # Hyper-paramters
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.proj_dim = 256
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_ctgs = args.num_labeled_classes + args.num_unlabeled_classes

    args.num_domains = 2 

    # ----------------------
    # BASE MODEL
    # ----------------------
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)
    
    print(f'Loading weights from {pretrain_path}')
    state_dict = torch.load(pretrain_path, map_location='cpu')
    model.load_state_dict(state_dict,strict=False)

    # ----------------------
    # OPTIMIZATION
    # ----------------------
    params_groups = get_params_groups(model) #+ get_params_groups(sem_projector) + get_params_groups(dom_projector)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    args.fp16_scaler = None
    if args.fp16:
        args.fp16_scaler = torch.cuda.amp.GradScaler()

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    # CONTRASTIVE TRANSFORM
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    # DATASETS
    if args.task_type == 'A_L+A_U->B':
        train_dataset, unlabeled_dataset_A, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    elif args.task_type == 'A_L+A_U+B->A_U+B+C':
        train_dataset, unlabeled_dataset_A, unlabeled_dataset_B, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)


    if args.task_type == 'A_L+A_U+B->A_U+B+C':                                  
        test_loader_B = DataLoader(unlabeled_dataset_B, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=False)

    args.only_test = True
    test_loaders = []

    if args.dataset_name == 'domainnet':
        ovr_envs.remove(args.src_env)
        if args.task_type == 'A_L+A_U+B->A_U+B+C':
            ovr_envs.remove(args.aux_env)

        for d in ovr_envs:
            args.tgt_env = d
            test_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
            test_loader_C = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
            test_loaders.append(test_loader_C)
    
    else:
        for d in distortions:
            for s in severity:
                args.distortion, args.severity = d, str(s)
                test_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
                test_loader_C = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
                test_loaders.append(test_loader_C)


    # ----------------------
    # TRAIN
    # ----------------------
    best_test_acc, best_train_ul_acc, best_train2_ul_acc = 0, 0, 0
    best_train_ul_all_acc, best_train_ul_old_acc, best_train_ul_new_acc= 0, 0, 0
    best_train2_ul_all_acc, best_train2_ul_old_acc, best_train2_ul_new_acc= 0, 0, 0
    for epoch in range(args.epochs):
        print("Epoch: " + str(epoch))

        label_len = len(train_dataset.labelled_dataset) 
        unlabelled_len_1 = len(train_dataset.unlabelled_dataset1)
        unlabelled_len_2 = len(train_dataset.unlabelled_dataset2)

        total_len = label_len + unlabelled_len_1 + unlabelled_len_2
        weight_label = 1.0
        weight_unlabel_1 = label_len / unlabelled_len_1
        weight_unlabel_2 = label_len / unlabelled_len_2
        
        # 为每个样本分配对应的权重
        sample_weights = []
        for i in range(total_len):
            if i < label_len:
                sample_weights.append(weight_label)
            elif i < label_len + unlabelled_len_1:
                sample_weights.append(weight_unlabel_1)
            else:
                sample_weights.append(weight_unlabel_2)
                
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
        # --------------------
        # DATALOADERS
        # --------------------
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
        test_loader_A = DataLoader(unlabeled_dataset_A, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

        train(model, train_loader, optimizer, exp_lr_scheduler, cluster_criterion, epoch, args)
    
        if epoch % args.eval_freq == 0:
            with torch.no_grad():

                # Testing on unlabelled examples in domain A
                all_acc, old_acc, new_acc = test(model, test_loader_A, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
                if all_acc > best_train_ul_all_acc:
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
                    torch.save(model.state_dict(), os.path.join(args.model_path, 'dinoB16_best_trainul.pt'))
#                     best_train_ul_all_acc = all_acc
                    
                    best_train_ul_all_acc, best_train_ul_old_acc, best_train_ul_new_acc= all_acc, old_acc, new_acc
                    
#                 if args.dataset_name == 'domainnet':
                # Testing on unlabelled examples in domain B, if domain B exists
                if args.task_type == 'A_L+A_U+B->A_U+B+C':                                  
                    all_acc_B, old_acc_B, new_acc_B = test(model, test_loader_B, epoch=epoch, save_name='Train ACC Unlabelled2', args=args)
                    if all_acc_B > best_train2_ul_all_acc:
                        print('Best Train-2 Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_B, old_acc_B, new_acc_B))
                        torch.save(model.state_dict(), os.path.join(args.model_path, 'dinoB16_best_trainul2.pt'))
#                         best_train2_ul_all_acc = all_acc_B
                        best_train2_ul_all_acc, best_train2_ul_old_acc, best_train2_ul_new_acc= all_acc_B, old_acc_B, new_acc_B
    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(best_train_ul_all_acc, best_train_ul_old_acc, best_train_ul_new_acc))
    print('Best Train-2 Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(best_train2_ul_all_acc, best_train2_ul_old_acc, best_train2_ul_new_acc))
                # Testing on all examples in domain C
                all_acc_test_arr, old_acc_test_arr, new_acc_test_arr = [], [], []

                for test_loader in test_loaders: 
                    tmp_all_acc_test, tmp_old_acc_test, tmp_new_acc_test = test(model, test_loader, epoch=epoch, save_name='Test ACC', args=args)
                    all_acc_test_arr.append(tmp_all_acc_test)
                    old_acc_test_arr.append(tmp_old_acc_test)
                    new_acc_test_arr.append(tmp_new_acc_test)

                all_acc_test, old_acc_test, new_acc_test = sum(all_acc_test_arr)/len(test_loaders), sum(old_acc_test_arr)/len(test_loaders), sum(new_acc_test_arr)/len(test_loaders)

                if all_acc_test > best_test_acc:
                    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
                    print('Best Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

                    if args.dataset_name == 'domainnet':
                        for i, d in enumerate(ovr_envs):
                            print('For '+d+', All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test_arr[i], old_acc_test_arr[i], new_acc_test_arr[i]))
                        print('################################')

                    else:
                        col = len(severity)
                        for i, d in enumerate(distortions):
                            print('For '+d+', All {:.4f} | Old {:.4f} | New {:.4f}'.format(sum(all_acc_test_arr[i*col:(i+1)*col])/col, sum(old_acc_test_arr[i*col:(i+1)*col])/col, sum(new_acc_test_arr[i*col:(i+1)*col])/col))
                        print('################################')

                    torch.save(model.state_dict(), os.path.join(args.model_path, 'dinoB16_best.pt'))
                    best_test_acc = all_acc_test