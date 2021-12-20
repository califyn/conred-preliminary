# python imports
import os
import gc
import subprocess
import traceback
import argparse
import datetime
from tqdm.contrib import tenumerate
from tqdm import tqdm
import time
import random
from PIL import Image
import pickle
import json
from copy import deepcopy
from itertools import chain
from sklearn.linear_model import LogisticRegression
import sys
import tempfile

# sci suite
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ellipj
from scipy import stats
import sklearn
from sklearn import ensemble
from sklearn import multioutput
from sklearn.metrics import roc_auc_score

# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP

# local imports
import datasets # TODO: make sure this import actually works
import model
from tabtransformer.tabtransformer.tab_transformer_pytorch import CombTabTransformer
from sync_batchnorm.sync_batchnorm import convert_model

verbose = False

def most_recent_file(folder, ext=""):
    max_time = 0
    max_file = ""
    for dirname, subdirs, files in os.walk(folder):
        for fname in files:
            full_path = os.path.join(dirname, fname)
            time = os.stat(full_path).st_mtime
            if time > max_time and full_path.endswith(ext):
                max_time = time
                max_file = full_path

    return max_file

def string_to_list(inn):
    if len(inn) == 0:
        return []
    
    inn = inn.replace('(', '["')
    inn = inn.replace(')', '"]')
    inn = inn.replace(',', '","')
    
    if not inn.startswith("["):
        inn = '["' + inn
    if not inn.endswith("]"):
        inn = inn + '"]'

    return eval(inn)

def int_splitter(floats, my_sum):
    try:
        assert(abs(1 - sum(floats)) < 0.01)
    except:
        raise ValueError("splits must sum to 1!")
    orig_sum = my_sum

    tot_size = 0
    ret = [int(floats[0] * my_sum)]
    tot_size += ret[0]

    for i in range(1, len(floats) - 1):
        floats.pop(0)
        ret.append(int(floats[0]  * orig_sum))
        tot_size += ret[-1]
    ret.append(orig_sum - tot_size)

    return ret

log_file = None
def pwint(*args, sep=None):
    # printsand writes at the sametime
    print(*args)
    args = [str(x) for x in args]
    if sep == None:
        sep = " "
    args = sep.join(args)
    with open(log_file, 'a') as f:
        f.write(args + "\n")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class LRScheduler(object):
    """
    Learning rate scheduler for the optimizer.

    Warmup increases to base linearly, while base decays to final using cosine.
    """

    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False, flat_warmup=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        if not flat_warmup:
            warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        else:
            warmup_lr_schedule = np.linspace(warmup_lr, warmup_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr

def euclidean_dist(z1, z2):
    n = z1.shape[0]
    return 2 * z1 @ z2.T - torch.diagonal(z1 @ z1.T).repeat(n).reshape((n,n)).T - torch.diagonal(z2 @ z2.T).repeat(n).reshape((n,n))

def simsiam_loss(p, z, distance="cosine"):
    """
    Negative cosine similarity. (Cosine similarity is the cosine of the angle
    between two vectors of arbitrary length.)

    Contrastive learning loss with only *positive* terms.
    :param p: the first vector. p stands for prediction, as in BYOL and SimSiam
    :param z: the second vector. z stands for representation
    :return: -cosine_similarity(p, z)
    """
    if distance == "euclidean":
        return - torch.diagonal(euclidean_dist(p, z.detach())).mean()
    elif distance == "cosine":
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

def _uni_info_nce(z1, z2, temperature=0.1, distance="cosine", both_sides=True, remove_duplicates=False, targets=None, lam=None, perm_idx=None):
    """
    Noise contrastive estimation loss.
    Contrastive learning loss with *both* positive and negative terms.
    Note z1 and z2 must have the same dimensionality.
    :param z1: first vector
    :param z2: second vector
    :param temperature: how sharp the prediction task is
    :param both_sides: whether to use both-sided infoNCE
    :return: infoNCE(z1, z2)
    """
    if z1.size()[1] <= 1 and distance == "cosine":
        raise UserWarning('InfoNCE loss has only one dimension, add more dimensions')
    if remove_duplicates and targets != None:
        raise ValueError("don't do both remove duplicates & targets for clip acc")

    if remove_duplicates:
        z2_ = torch.unique(z2, dim=0)
        map_idx = [-1000] * len(z2)
        for i in range(len(z2_)):
            mask = [torch.equal(x, z2_[i]) for x in z2]
            mask = [it for it, elem in enumerate(mask) if elem]
            for m in mask:
                map_idx[m] = i
        z2 = deepcopy(z2_.detach())

    if both_sides:
        combined = torch.cat((z1, z2))
        z1 = combined
        z2 = combined

    if distance == "cosine":
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        logits = z1 @ z2.T
    elif distance == "euclidean":
        logits = euclidean_dist(z1, z2)

    logits /= temperature
    if torch.cuda.is_available(): # TODO: add projectors, also make it symmetric
        logits = logits.cuda()

    if both_sides:
        n = z1.shape[0] // 2
    else:
        n = z1.shape[0]
    if targets != None:
        labels = targets
    elif both_sides:
        labels = torch.arange(0, 2 * n, dtype=torch.long)
        #labels = labels + (1 - labels % 2)
        labels[:n] = labels[:n] + n
        labels[n:] = labels[n:] - n
        labels = labels.tolist()
    else:
        labels = torch.arange(0, n, dtype=torch.long).tolist()

    if remove_duplicates:
        labels = [map_idx[i] for i in labels]

    labels = torch.LongTensor(labels).cuda()

    if torch.cuda.is_available():
        labels = labels.cuda()

    if lam != None:
        loss = (1 - lam) * torch.nn.functional.cross_entropy(logits, labels) + lam * torch.nn.functional.cross_entropy(logits, perm_idx) 
    else:
        loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def info_nce(z, temperature=0.1, distance="cosine", both_sides=True, lam=None, perm_idx=None, twoway=False, fourway=False, lr_weight=[1,1]):
    # wrapper to do infonce on multiple contrastive objectives
    loss = []
    if not twoway and not fourway:
        for it1, z1 in enumerate(z):
            for it2, z2 in enumerate(z):
                if it1 == it2:
                    continue
                
                loss.append(torch.unsqueeze(_uni_info_nce(z1, z2, temperature, distance, both_sides), dim=0))
    elif twoway:
        for it, z2 in enumerate(z[1:]):
            loss.append(torch.unsqueeze(_uni_info_nce(z[0], z2, temperature, distance, both_sides=False, lam=lam, perm_idx=perm_idx), dim=0) * lr_weight[it])
    elif fourway:
        for it, z2 in enumerate(z[1:]):
            loss.append(torch.unsqueeze(_uni_info_nce(z[0], z2, temperature, distance, both_sides=False, lam=lam, perm_idx=perm_idx), dim=0) * lr_weight[it])
            loss.append(torch.unsqueeze(_uni_info_nce(z2, z[0], temperature, distance, both_sides=False, lam=lam, perm_idx=perm_idx), dim=0) * lr_weight[it])
    loss = torch.mean(torch.cat(loss))

    return loss

def confusion_matrix_str(cm, normalize=True, figs=3, label_names=None):
    if not normalize:
        raise NotImplementedError("non normalized arrays not available yet")
    else: 
        cm = cm / np.sum(cm, axis=0, keepdims=True)

    n = np.size(cm, axis=0)
    #color_scale = ["\u001b[38;5;$17m", "\u001b[38;5;$60m", "\u001b[38;5;$109m", "\u001b[38;5;$137m", "\u001b[38;5;$167m", "\u001b[38;5;$196m"]
    color_scale = ["", "", "", "", "", ""]
    #bold = "\033[1m"
    #reset = "\u001b[0m"
    bold = ""
    reset = ""
    bounds = np.array([0, 0.2, 0.4, 0.6, 0.8])

    ret = ""
    ret += (" ") * (figs + 1)
    for i in range(1, n + 1):
        ret += " "
        if label_names == None:
            ret += '{:>{width}}'.format(str(i), width=figs + 1)
        else:
            ret += '{:>{width}}'.format(label_names[i - 1], width=figs + 1)
    ret += "\n"

    for i in range(1, n + 1):
        if label_names == None:
            ret += '{:>{width}}'.format(str(i), width=figs + 1)
        else:
            ret += '{:>{width}}'.format(label_names[i - 1], width=figs + 1)

        for j in range(1, n + 1):
            ret += " "
            ret += color_scale[np.searchsorted(bounds, cm[i - 1, j - 1])]
            if i == j:
                ret += bold
            if str(cm[i - 1, j - 1]) == "nan":
                ret += "nan".rjust(figs + 1)
            elif cm[i - 1, j - 1] >= 1:
                ret += '{:.{width}f}'.format(cm[i - 1, j - 1], width=figs - 1)
            else:
                ret += '{:.{width}f}'.format(cm[i - 1, j - 1], width=figs)[1:]
            ret += reset
        ret += "\n"

    return ret

def clip_acc(z1, z2, distance="cosine", as_confusion_matrix=False, labels=None, label_size=-1, remove_duplicates=False, targets=None):
    """
    CLIP classification accuracy objective.
    The task is vaguely matching each z1 to z2.
    :param z1: left outputs to be matched
    :param z2: right outputs (targets)
    :param distance: distance metric to use
    :param targets: indices of targets for z1
    """
    if remove_duplicates and targets != None:
        raise ValueError("don't do both remove duplicates & targets for clip acc")

    if remove_duplicates:
        z2_ = torch.unique(z2, dim=0)
        map_idx = [-1000] * len(z2)
        for i in range(len(z2_)):
            mask = [torch.equal(x, z2_[i]) for x in z2]
            mask = [it for it, elem in enumerate(mask) if elem]
            for m in mask:
                map_idx[m] = i
        z2 = z2_

    if distance == "cosine":
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        dists = torch.matmul(z1, z2.T) 
    elif distance == "euclidean":
        dists = euclidean_dist(z1, z2)

    n = z1.shape[0]
    if targets != None:
        default_targets = False
    else:
        default_targets = True
        targets = list(range(n))

    if remove_duplicates:
        targets = [map_idx[i] for i in targets]

    targets = torch.LongTensor(targets).cuda()

    if not as_confusion_matrix:
        neighbors = torch.argmax(dists, dim=1) 
        neighbors = neighbors - targets 

        return torch.numel(torch.where(neighbors == 0)[0]) / n
    else:
        if not default_targets:
            raise ValueError("Confusion matrix not implemented for nl targets")

        neighbors = torch.argmax(dists, dim=1) 
        if label_size == -1:
            label_size = int(max(labels)) + 1
        pairs = torch.stack((neighbors, targets), dim=1)
        pairs = pairs.detach().tolist()
        pairs = [[int(labels[x].item()) for x in l] for l in pairs]
        cm = np.zeros((label_size, label_size))
        for pair in pairs:
            cm[pair[0], pair[1]] = cm[pair[0], pair[1]] + 1

        if default_targets:
            class_sizes = []
            class_accs = []
            for i in range(label_size):
                class_sizes.append(torch.numel(torch.where(labels == i)[0]))

                if class_sizes[-1] == 0:
                    class_accs.append(float('nan'))
                    continue

                dists_mini = dists[labels == i, :]
                dists_mini = dists_mini[:, labels == i]
                dists_mini = dists_mini.reshape((class_sizes[-1], class_sizes[-1]))
                nb_mini = torch.argmax(dists_mini, dim=1)

                nb_mini = nb_mini - torch.arange(0, class_sizes[-1]).cuda()
                class_accs.append(torch.numel(torch.where(nb_mini == 0)[0]) / class_sizes[-1]) 
            # correct = [0] * label_size
            # total = [0] * label_size
            # for i in range(label_size):
            #     for it, pair in enumerate(pairs):
            #         if pair == [i, i]:
            #             if neighbors[it] == targets[it]:
            #                 correct[i] = correct[i] + 1
            #             total[i] = total[i] + 1
            return class_accs, class_sizes, cm
        else:
            return cm

def validate(args, encoder, datahandler):
    encoder = evaluate_single(args, [encoder], datahandler, -1, clip_inv=False) 
    return encoder

    # dataset
    # dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     shuffle=True,
    #     batch_size=args.bsz,
    #     **dataloader_kwargs
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     shuffle=True,
    #     batch_size=args.bsz,
    #     **dataloader_kwargs
    # )

    # if verbose:
    #     print("[Validation] Completed data loading")


    # # optimization
    # optimizers = []
    # if isinstance(encoder, model.TransformerWithMLP):
    #     trns_optimizer = torch.optim.AdamW(encoder.trns.parameters(), lr=args.l_lr[0], weight_decay=args.wd)
    #     mlp_optimizer = torch.optim.SGD(encoder.mlp.parameters(), lr=args.l_lr[1], weight_decay=args.wd, momentum=args.p) # TODO: should i use two different optimizers? or just one
    #     
    #     optimizers += [trns_optimizer, mlp_optimizer]
    # elif isinstance(encoder, model.SimpleMLP):
    #     mlp_optimizer = torch.optim.SGD(encoder.parameters(), lr=args.g_lr, momentum=args.p)

    #     optimizers += [mlp_optimizer]
    # input(optimizers)
    # encoder = torch.nn.DataParallel(encoder)
    # #lr_scheduler = LRScheduler(
    # #    optimizer=optimizer,
    # #    warmup_epochs=args.warmup_epochs,
    # #    warmup_lr=0,
    # #    num_epochs=args.epochs,
    # #    base_lr=args.lr * args.bsz / 256,
    # #    final_lr=0,
    # #    iter_per_epoch=len(train_loader),
    # #    constant_predictor_lr=True
    # #)

    # if verbose:
    #     print("[Validation] Model generation complete, training begins")

    # # logging
    # start = time.time()
    # os.makedirs(args.path_dir, exist_ok=True)
    # torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, '0.pth'))

    # data_args = vars(args)

    # is_classification = train_dataset[:]["target"]
    # is_classification = torch.all((is_classification - torch.round(is_classification)) == 0)
    # if is_classification:
    #     loss_type = "cross_entropy"
    # else:
    #     loss_type = "mean_square"

    # with open(os.path.join(args.path_dir, '0.args'), 'w') as fp:
    #     json.dump(data_args, fp, indent=4)

    # saved_loss = "None"
    # saved_var = "None"

    # for e in range(1, args.epochs + 1):
    #     if e % args.progress_every == 0:
    #         with torch.no_grad():
    #             print("epoch ", e, "!")
    #             print("loss: ", saved_loss)
    #             print("var: ", saved_var)
    #             acc = 0
    #             total = 0
    #             for it, elem in enumerate(test_loader):
    #                 out = torch.argmax(encoder(elem["left"]), dim=1)

    #                 if loss_type == "cross_entropy":
    #                     to_add = torch.where(out == elem["target"])[0].size()
    #                 elif loss_type == "mean_square":
    #                     to_add = torch.where(out == torch.argmax(elem["target"], dim=1))[0].size()

    #                 if len(to_add) == 0:
    #                     to_add = 1
    #                 else:
    #                     to_add = to_add[0]
    #                 acc += to_add
    #                 total += elem["target"].size()[0]
    #             print("accuracy: ", acc / total)
    #     
    #     encoder.train()

    #     temp_saved_loss = []
    #     for it, elem in tenumerate(train_loader):
    #         encoder.zero_grad()

    #         if loss_type == "cross_entropy":
    #             loss = torch.nn.CrossEntropyLoss()
    #             out = encoder(elem["left"])
    #             saved_var = torch.mean(torch.std(out, dim=0))
    #             l = loss(out, elem["target"].long())
    #         elif loss_type == "mean_square":
    #             loss = torch.nn.MSELoss()
    #             out = encoder(elem["left"])
    #             saved_var = torch.mean(torch.std(out, dim=0))
    #             l = loss(out, elem["target"])
    #         temp_saved_loss.append(float(l.item()))

    #         l.backward()
    #         for optimizer in optimizers:
    #             optimizer.step()
    #     saved_loss = torch.mean(torch.Tensor(temp_saved_loss))

    #     if e % args.save_every == 0:
    #         torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, f'{e}.pth'))

    #         with open(os.path.join(args.path_dir, f'{e}.args'), 'w') as fp:
    #             json.dump(data_args, fp, indent=4)
    #         print("[saved]")

    # if verbose:
    #     print("[Validation] Training complete")

    # return encoder

def evaluate_single(args, encoders, datahandler, save_num, clip_inv=True, tasks=[-1], encoder_idx=0, finetune=True, expensive=False):
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)

    if tasks == [-1]:
        tasks = args.zero_shot + args.finetune

    if clip_inv:
        if verbose:
            pwint("[Evaluate] begin CLIP inverse evaluation...")

        for encoder in encoders:
            encoder.eval()

        val_loader = torch.utils.data.DataLoader(
                dataset=datahandler.clip_test,
                shuffle=True,
                batch_size=args.bsz,
                **dataloader_kwargs)

        for oit, dtype in enumerate(args.contrastive):
            if oit == encoder_idx:
                continue

            acc = [] 
            loss = []
            cms = []
            cms_acc = []
            cms_total = []
            if expensive:
                pcs_acc = []
                pcs_total = []
                pcs_manual = []
                pcs_mtotal = []
                sites_acc = []
                sites_total = []

            distance = "cosine"
            if args.euclidean:
                distance = "euclidean"
           
            if expensive:
                num_rep = args.type_reps
            else:
                num_rep = 1
            for _ in tqdm(range(num_rep)):
                #for it, elem in tenumerate(val_loader):
                for it, elem in enumerate(val_loader):
                    with torch.no_grad():
                        base_out = encoders[encoder_idx](elem[args.contrastive[encoder_idx]])
                        comp_out = encoders[oit](elem[dtype])

                        acc.append(clip_acc(base_out, comp_out, distance=distance))
                        loss.append(info_nce([base_out, comp_out], distance=distance).item())
                        #_, _, cm = clip_acc(base_out, comp_out, distance=distance, as_confusion_matrix=True, labels=elem["type"], label_size=datahandler.num_types)
                        pc1, pc2, cm = clip_acc(base_out, comp_out, distance=distance, as_confusion_matrix=True, labels=elem["type"], label_size=datahandler.num_types)
                        
                    cms.append(cm)
                    cms_acc.append(pc1)
                    cms_total.append(pc2)

            if expensive:
                if dtype == "reports":
                    type_loader = datahandler.by_type(args.type_bsz, select_size=-1, dataset='test', reps=args.type_reps) 
                else:
                    type_loader = datahandler.by_type(args.type_bsz, select_size=-1, dataset='test', reps=1) 
                overpredict_rat = []
                overpredict_adj = []
                for it, elem in tenumerate(type_loader, total=args.type_reps * datahandler.num_types if dtype == "reports" else datahandler.num_types):
                    with torch.no_grad():
                        base_out = encoders[encoder_idx](elem[args.contrastive[encoder_idx]])
                        comp_out = encoders[oit](elem[dtype])

                        if dtype == "reports" and args.compare_nl:
                            with open("compare_nl.txt", "a") as f:
                                y_true = elem["reports"]["input_ids"]
                                y_true_id = elem["id"][0]
                                y_pred = torch.nn.functional.normalize(base_out, dim=1) @ torch.nn.functional.normalize(comp_out, dim=1).T
                                y_pred = torch.argmax(y_pred, axis=1)
                                y_pred_id = [elem["id"][0][x] for x in y_pred.tolist()]
                                y_pred = y_true[y_pred]
                                
                                y_true_site = [x[5:7] for x in y_true_id]
                                y_pred_site = [x[5:7] for x in y_pred_id]
                                #print(f"sites {sorted(list(set(y_true_site)))}")
                                #print(f"y true {y_true_site}")
                                #print(f"y pred {y_pred_site}")
                                #print(f"num matches {len([it for it in range(len(y_true_site)) if y_true_site[it] == y_pred_site[it]])}")
                                #print(f"num bad matches {len([it for it in range(len(y_true_site)) if y_true_site[it] == y_pred_site[it] and y_true_id[it] != y_pred_id[it]])}")
                                overpredict_rat.append(len([it for it in range(len(y_true_site)) if y_true_site[it] == y_pred_site[it]]) / len(y_true) * len(list(set(y_true_site))))
                                overpredict_adj.append(len([it for it in range(len(y_true_site)) if y_true_site[it] == y_pred_site[it] and y_true_id[it] != y_pred_id[it]]) / len(y_true) * len(list(set(y_true_site))))

                                y_true = [datahandler.tokenizer.convert_ids_to_tokens(w) for w in y_true]
                                y_pred = [datahandler.tokenizer.convert_ids_to_tokens(w) for w in y_pred]
                                y_true = [datahandler.tokenizer.convert_tokens_to_string(w) for w in y_true]
                                y_pred = [datahandler.tokenizer.convert_tokens_to_string(w) for w in y_pred]
                                y_true = [" ".join([l for l in w.split() if "[" not in l]) for w in y_true]
                                y_pred = [" ".join([l for l in w.split() if "[" not in l]) for w in y_pred]
                                f.write(str(save_num) + "\n")
                                f.write(str(datahandler.nl_type_map[it % len(datahandler.nl_type_map)]) + "\n")
                                f.write("\n".join([str((a, b, c, d)) for a, b, c, d in zip(y_true_id, y_true, y_pred_id, y_pred)]))
                                f.write("\n\n")

                        site_elem = [x[5:7] for x in elem["id"][0]]
                        site_set = list(set(site_elem))
                        si1, si2, _ = clip_acc(base_out, comp_out, distance=distance, as_confusion_matrix=True, labels=torch.Tensor([site_set.index(x) for x in site_elem]).cuda(), label_size=len(site_set))
                        pc1, pc2, _ = clip_acc(base_out, comp_out, distance=distance, as_confusion_matrix=True, labels=elem["type"], label_size=datahandler.num_types)
                    pcs_acc.append(pc1)
                    pcs_total.append(pc2)
                    sites_acc = sites_acc + si1
                    sites_total = sites_total + si2

                if dtype == "reports" and args.compare_nl:
                    for ctype in args.manual:
                        manual_loader = datahandler.manual_loader(args.type_bsz, reps=args.type_reps, type=ctype) 
                        
                        for it, elem in tenumerate(manual_loader):
                            with torch.no_grad():
                                base_out = encoders[encoder_idx](elem[args.contrastive[encoder_idx]])
                                comp_out = encoders[oit](elem["manual-" + dtype])

                                with open("compare_nl_manual.txt", "a") as f:
                                    y_true = elem["manual-reports"]["input_ids"]
                                    y_pred = torch.nn.functional.normalize(base_out, dim=1) @ torch.nn.functional.normalize(comp_out, dim=1).T
                                    y_pred = torch.argmax(y_pred, axis=1)
                                    y_pred = y_true[y_pred]

                                    y_true = [datahandler.tokenizer.convert_ids_to_tokens(w) for w in y_true]
                                    y_pred = [datahandler.tokenizer.convert_ids_to_tokens(w) for w in y_pred]
                                    y_true = [datahandler.tokenizer.convert_tokens_to_string(w) for w in y_true]
                                    y_pred = [datahandler.tokenizer.convert_tokens_to_string(w) for w in y_pred]
                                    y_true = [" ".join([l for l in w.split() if "[" not in l]) for w in y_true]
                                    y_pred = [" ".join([l for l in w.split() if "[" not in l]) for w in y_pred]
                                    f.write(str(save_num) + "\n")
                                    f.write(str(it) + "\n")
                                    f.write("\n".join([str((x, y)) for x, y in zip(y_true, y_pred)]))
                                    f.write("\n\n")

                                pc1, pc2, _ = clip_acc(base_out, comp_out, distance=distance, as_confusion_matrix=True, labels=elem["type"], label_size=datahandler.num_types)
                            pcs_manual.append(pc1)
                            pcs_mtotal.append(pc2)

            cms = np.sum(np.array(cms), axis=0)
            cms_acc = np.nanmean(np.array(cms_acc), axis=0)
            cms_sample = np.nanmean(np.array(cms_total), axis=0)
            loss = np.array(loss)
            if expensive:
                pcs_acc = np.nanmean(np.array(pcs_acc), axis=0)
                pcs_sample = np.reciprocal(np.sum(np.reciprocal(np.array(pcs_total)), axis=0)) / args.type_reps
                pcs_manual = np.nanmean(np.array(pcs_manual), axis=0)
                pcs_msample = np.sum(np.array(pcs_mtotal), axis=0) / args.type_reps
                print(np.bincount(np.array(sites_total)), "sites")
                sites_rat = np.array([sites_acc[it] * sites_total[it] for it in range(len(sites_total)) if sites_total[it] == 2])

            pwint(f"Data type {dtype}")
            pwint("Val acc:", np.around(acc, 3), np.around(np.mean(acc), 3))
            pwint("Val loss:", np.around(loss, 3), np.around(np.mean(loss), 3))
            pwint("Confusion Matrix:", confusion_matrix_str(cms, label_names=datahandler.nl_type_map), sep='\n')
            pwint("Indices | Within-Class Acc. | Mean Class Size | Model Class Size")
            pwint("".join('{:>6}'.format(datahandler.nl_type_map[i]) for i in range(0, datahandler.num_types))[1:])
            pwint("".join('{:>6}'.format(str(x)) for x in list(np.around(cms_acc * 100, 1)))[1:])
            pwint("".join('{:>6}'.format(str(x)) for x in list(np.around(cms_sample, 1)))[1:])
            pwint("".join('{:>6}'.format(str(x)) for x in list(np.around(1 / cms_acc, 1)))[1:])
            if expensive:
                pwint("Indices | Within-Class Acc. | Mean Class Size | Model Class Size")
                pwint("".join('{:>6}'.format(datahandler.nl_type_map[i]) for i in range(0, datahandler.num_types))[1:])
                pwint("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_acc * 100, 1)))[1:])
                pwint("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_sample, 1)))[1:])
                pwint("".join('{:>6}'.format(str(x)) for x in list(np.around(1 / pcs_acc, 1)))[1:])
                pwint(pcs_manual, "manual")
                pwint(pcs_msample, "sample")
                pwint(stats.describe(sites_rat), "sites prediction")
                if dtype == "reports":
                    pwint(np.mean(np.array(overpredict_rat)), "overprediction of same site")
                    pwint(np.mean(np.array(overpredict_adj)), "overprediction of same site (adjusted)")

            # file_to_update.write(f"Data type {dtype}")
            # file_to_update.write(f"Val acc: {np.around(acc, 3)} | {np.around(np.mean(acc), 3)}")
            # file_to_update.write(f"Val loss: {np.around(loss, 3)} | {np.around(np.mean(loss), 3)}")
            # file_to_update.write(f"Confusion matrix: \n {confusion_matrix_str(cms)}")
            # if expensive:
            #     file_to_update.write("Indices | Within-Class Acc. | Mean Class Size | Model Class Size")
            #     file_to_update.write("".join('{:>6}'.format(datahandler.nl_type_map[i]) for i in range(0, datahandler.num_types))[1:])
            #     file_to_update.write("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_acc * 100, 1)))[1:])
            #     file_to_update.write("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_sample, 1)))[1:])
            #     file_to_update.write("".join('{:>6}'.format(str(x)) for x in list(np.around(1 / pcs_acc, 1)))[1:])
            #     file_to_update.write(f"{pcs_manual} manual")
            #     file_to_update.write(f"{pcs_msample} sample")
            # file_to_update.flush()

    ft_encoders = []
    if expensive and len(tasks) > 0:
        pwint(f"[Evaluate] begin evaluation on tasks {tasks}")

        for it, task in enumerate(tasks):
            #if finetune:
            pwint(f"[Evaluate] begin task {task} evaluation ({it} out of {len(tasks)})")

            train_loader = torch.utils.data.DataLoader(
                    dataset=datahandler.val_train[task],
                    shuffle=True,
                    batch_size=args.val_bsz,
                    **dataloader_kwargs)
            test_loader = torch.utils.data.DataLoader(
                    dataset=datahandler.val_test[task],
                    shuffle=True,
                    batch_size=args.val_bsz,
                    **dataloader_kwargs)

            if "_nl" not in task:
                pass
                
                # output_nclasses = int(max(torch.max(datahandler.val_train[task][:][task]).item(), torch.max(datahandler.val_test[task][:][task]).item()) + 1)

                # ft_encoder = model.WithFinetuneLayers(encoders[encoder_idx], args.repr_dim, output_size=output_nclasses, size=[])
                # ft_encoders.append(ft_encoder)

                # if finetune:
                #     model_optimizer = torch.optim.AdamW(ft_encoder.model.parameters(), lr=args.ft_lr[0], weight_decay=args.wd)
                # ft_optimizer = torch.optim.AdamW(ft_encoder.finetune.parameters(), lr=args.ft_lr[1], weight_decay=args.wd)

                # for e in range(1, 2 + args.ft_epochs):
                #     # evaluate
                #     ft_encoder.eval()

                #     correct = 0
                #     total = 0
                #     saved_loss = []
                #     with torch.no_grad():
                #         for it, elem in enumerate(test_loader):
                #             out = ft_encoder(elem[args.contrastive[encoder_idx]])

                #             if output_nclasses != 2:
                #                 loss = torch.nn.CrossEntropyLoss()
                #                 l = loss(out, elem[task].long())
                #             else:
                #                 loss = torch.nn.BCEWithLogitsLoss()
                #                 l = loss(out, torch.nn.functional.one_hot(elem[task].long(), num_classes=output_nclasses).float())
                #             saved_loss.append(float(l.item()))

                #             out = torch.argmax(out, dim=1)
                #             correct += torch.numel(torch.where(out == elem[task])[0])
                #             total += int(elem[list(elem.keys())[0]].size()[0])
                #     pwint(f"[Evaluate] on epoch {e - 1}, task {task}: accuracy {round(correct/total, 3)}")
                #     # file_to_update.write(f"epoch {e - 1} & task {task} & zero shot: accuracy {round(correct/total, 3)}")
                #     pwint(f"test loss: {np.mean(np.array(saved_loss))}")

                #     #if e > args.ft_epochs:
                #     #    if save_num != -1:
                #     #        torch.save(dict(epoch=0, state_dict=ft_encoder.state_dict()), os.path.join(args.path_dir, str(save_num) + "-" + task + ".pth"))
                #     #    break

                #     # train
                #     ft_encoder.train()

                #     pwint(f"[Evaluate] epoch: {e} out of {args.ft_epochs}")
                #     saved_loss = []
                #     for it, elem in enumerate(train_loader):
                #         out = ft_encoder(elem[args.contrastive[encoder_idx]]) 

                #         if output_nclasses != 2:
                #             loss = torch.nn.CrossEntropyLoss()
                #             l = loss(out, elem[task].long())
                #         else:
                #             loss = torch.nn.BCEWithLogitsLoss()
                #             l = loss(out, torch.nn.functional.one_hot(elem[task].long(), num_classes=output_nclasses).float())
                #         saved_loss.append(float(l.detach().item()))

                #         l.backward()
                #         if finetune:
                #             model_optimizer.step()
                #         ft_optimizer.step()
                #     pwint(f"train loss: {np.mean(np.array(saved_loss))}")

            else:
                ft_encoder = deepcopy(encoders[encoder_idx].module)
                if "reports" in args.contrastive:
                    nl_encoder = encoders[args.contrastive.index("reports")].module
                elif "clean-reports" in args.contrastive:
                    nl_encoder = encoders[args.contrastive.index("clean-reports")].module
                else:
                    raise ValueError("nl target but can't find nl encoder")
                
                # ft_optimizer = torch.optim.AdamW(ft_encoder.parameters(), lr=args.ft_lr[0], weight_decay=args.wd)
                # nl_optimizer = torch.optim.AdamW(nl_encoder.parameters(), lr=args.ft_lr[0], weight_decay=args.wd)

                label_size = torch.numel(torch.unique(datahandler.val_train[task][:][task[:-3]]))

                for e in range(1, 2 + args.ft_epochs):
                    # evaluate
                    ft_encoder.eval()
                    nl_encoder.eval()

                    #acc = []
                    #cm = []
                    true = []
                    predict = []
                    ids = []
                    with torch.no_grad():
                        for it, elem in enumerate(test_loader):
                            ft_out = torch.unsqueeze(ft_encoder(elem[args.contrastive[encoder_idx]]), 1)
                            #nl_out = nl_encoder(elem["reports"])
                            #nl_out = nl_encoder(elem[task])
                            #nl_out = nl_encoder(datahandler.nl_prompts[task])
                            nl_out = torch.stack((
                                    nl_encoder({
                                        "input_ids": elem[task]["neg_input_ids"],
                                        "attention_mask": elem[task]["neg_attention_mask"]
                                    }),
                                    nl_encoder({
                                        "input_ids": elem[task]["pos_input_ids"],
                                        "attention_mask": elem[task]["pos_attention_mask"]
                                    })
                            ), dim=2)

                            #acc.append(clip_acc(ft_out, nl_out, distance=distance, remove_duplicates=True))
                            #cmmm = clip_acc(ft_out, nl_out, distance=distance, as_confusion_matrix=True, labels=elem[task[:-3]], label_size=label_size)[2]
                            #num_ones = torch.sum(elem[task[:-3]]) / torch.numel(elem[task[:-3]])
                            #acc.append(float((cmmm[0, 0] / (cmmm[0, 0] + cmmm[0, 1]) * (1 - num_ones) + cmmm[1, 1] / (cmmm[1, 1] + cmmm[1, 0])  * num_ones).item()))
                            #cm.append(cmmm) 
                            assert(distance == 'cosine')
                            ft_out = torch.nn.functional.normalize(ft_out, dim=2)
                            nl_out = torch.nn.functional.normalize(nl_out, dim=1)

                            #dist = ft_out @ nl_out.T
                            dist = torch.squeeze(torch.bmm(ft_out, nl_out), dim=1)
                            dist = 1 / (1 + torch.exp(dist[:, 0] - dist[:, 1])) # softmax
                            true = true + elem[task[:-3]].tolist()
                            predict = predict + dist.detach().tolist()
                            ids = ids + list(elem["id"][0])

                    #auroc.append(roc_auc_score(elem[task[:-3]], dist)
                    #pwint(np.array(true))
                    #pwint(np.array(predict))
                    # print(np.unique(true), "True")
                    # print(len(true), true[:50], "true shape")
                    # print(np.min(np.array(predict)), np.max(np.array(predict)), "pred")
                    # print(len(predict), predict[:50], "pred shape")
                    print("temporary fix;;;;;;; get rid of this asap")
                    true = np.array(true)
                    predict = np.array(predict)
                    ids = [id for it, id in enumerate(ids) if true[it] != -1]
                    predict = predict[true != -1]
                    true = true[true != -1]
                    auroc = roc_auc_score(np.array(true), np.array(predict))
                    pwint(f"[Evaluate]: nl auroc on task {task}: {round(float(auroc), 3)}")
                    pd.DataFrame({
                        task + "-ids": ids,
                        task + "-true": true,
                        task + "-pred": predict
                    }).to_csv(os.path.join(args.path_dir, f"{save_num}-{task}.csv")) 
                            
                    #acc = np.mean(np.array(acc))
                    #cm = np.array(cm)
                    #if len(cm.shape) > 2:
                    #    cm = np.mean(cm, axis=0)
                    #pwint(f"[Evaluate] on epoch {e - 1}, task {task}: accuracy {round(float(acc), 3)}")
                    #pwint(f"[Evaluate] cofusion matrix: {confusion_matrix_str(cm, label_names=[str(x) for x in range(label_size)])}")
                    # file_to_update.write(f"epoch {e - 1} & task {task} & zero shot: accuracy {round(float(acc), 3)}")

                    # if e > args.ft_epochs:
                        #if save_num != -1:
                        #    torch.save(dict(epoch=0, state_dict=ft_encoder.state_dict()), os.path.join(args.path_dir, str(save_num) + "-" + task + ".pth"))
                    break

                    # train
                    # ft_encoder.train()
                    # nl_encoder.train()

                    # pwint(f"[Evaluate] nl-epoch: {e} out of {args.ft_epochs}")
                    # for it, elem in enumerate(train_loader):
                    #     ft_out = ft_encoder(elem[args.contrastive[encoder_idx]])
                    #     nl_out = nl_encoder(elem[task])

                    #     l = _uni_info_nce(ft_out, nl_out, distance=distance, remove_duplicates=True, both_sides=False)

                    #     l.backward()
                    #     if finetune:
                    #         model_optimizer.step()
                    #     ft_optimizer.step()
            # else:
            #     output_nclasses = int(max(torch.max(datahandler.val_train[task][:][task]).item(), torch.max(datahandler.val_test[task][:][task]).item()) + 1)

            #     if output_nclasses != 2:
            #         raise ValueError("have not implemented non finetuning for more than 2 classes")

            #     train_outs = []
            #     train_targets = []
            #     test_outs = []
            #     test_targets = []
            #     
            #     train_loader = torch.utils.data.DataLoader(
            #             dataset=datahandler.val_train[task],
            #             shuffle=True,
            #             batch_size=args.val_bsz,
            #             **dataloader_kwargs)
            #     test_loader = torch.utils.data.DataLoader(
            #             dataset=datahandler.val_test[task],
            #             shuffle=True,
            #             batch_size=args.val_bsz // 4,
            #             **dataloader_kwargs)

            #     for it, elem in enumerate(train_loader):
            #         train_outs.append(torch.nn.functional.normalize(encoders[encoder_idx](elem[args.contrastive[encoder_idx]]).detach().cpu(), dim=1).numpy())
            #         train_targets.append(elem[task].detach().cpu().numpy())

            #     for it, elem in enumerate(test_loader):
            #         test_outs.append(torch.nn.functional.normalize(encoders[encoder_idx](elem[args.contrastive[encoder_idx]]).detach().cpu(), dim=1).numpy())
            #         test_targets.append(elem[task].detach().cpu().numpy())

            #     train_outs = np.array(train_outs)
            #     test_outs = np.array(test_outs)
            #     train_targets = np.array(train_targets)
            #     test_targets = np.array(test_targets)

            #     print(train_outs.shape)
            #     print(test_outs.shape)
            #     train_outs = np.reshape(train_outs, (np.size(train_outs, axis=0) * np.size(train_outs, axis=1), args.repr_dim))
            #     test_outs = np.reshape(test_outs, (np.size(test_outs, axis=0) * np.size(test_outs, axis=1), args.repr_dim))
            #     train_targets = np.reshape(train_targets, (-1, ))
            #     test_targets = np.reshape(test_targets, (-1, ))

            #     results = []
            #     
            #     for i in range(args.ft_trials):
            #         clf = LogisticRegression().fit(train_outs, train_targets)
            #         results.append(clf.score(test_outs, test_targets))
            #     results = np.array(results)
            #     #results = np.array([0, 0, 1])

            #     print(f"Logistic classifier accuracy on task {task}: {np.mean(results)}")
            #     file_to_update.write(f"Logistic classifier accuracy on task {task}: {np.mean(results)}")
            #     file_to_update.flush()
            #     torch.cuda.empty_cache()

    return ft_encoders

#datahandler = None

def write_all(args, encoders, datahandler):
    #columns = ['is_test'] + args.contrastive + [t[:-3] for t in args.finetune if t[-3:] == '_nl']
    columns = ['is_test']
    for c in args.contrastive:
        columns = columns + [c + '-' + str(i) for i in range(args.repr_dim)]
    for t in args.finetune:
        if t[-3:] == '_nl':
            #columns = columns + [t[:-3]] + [t + '-neg-' + str(i) for i in range(args.repr_dim)] + [t + '-pos-' + str(i) for i in range(args.repr_dim)]
            columns = columns + [t[:-3]]

    df = pd.DataFrame(columns=columns)

    for encoder in encoders:
        encoder.eval()

    train_loader = torch.utils.data.DataLoader(
            dataset=datahandler.pretrain,
            shuffle=False,
            batch_size=args.bsz,
            drop_last=False,
            pin_memory=False,
            num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
            dataset=datahandler.clip_test,
            shuffle=False,
            batch_size=args.bsz,
            drop_last=False,
            pin_memory=False,
            num_workers=0
    )

    with torch.no_grad():
        for (test_indicator, loader) in zip((False, True), (train_loader, test_loader)):
            for elem in loader:
                #temp_df = pd.DataFrame(columns=columns)
                temp_nd = np.zeros((len(elem["id"][0]), len(columns))).astype('float64')

                for it, c in enumerate(args.contrastive):
                    out = encoders[it](elem[c]).detach().cpu().numpy()
                    temp_nd[:, 1 + args.repr_dim * it:1 + args.repr_dim * (it + 1)] = out

                    #for it2, id in enumerate(elem["id"][0]):
                    #    df[id, c] = out[it2]

                # for it2, id in enumerate(elem["id"][0]):
                #     for task in finetune:
                #         df[id, task[:-3]] = out[finetune][it2]
                #     df[id, "id"] = id
                #     df[id, "is_test"] = test_indicator
                #temp_df["id"] = elem["id"][0]
                #temp_df["is_test"] = test_indicator
                temp_nd[:, 0] = test_indicator

                add_it = 1 + args.repr_dim * len(args.contrastive)
                for task in args.finetune:
                    if task[-3:] == "_nl":
                        temp_nd[:, add_it] = elem[task[:-3]].cpu().numpy()
                        # ft_out = torch.unsqueeze(encoders[args.contrastive.index("rna-seq")](elem["rna-seq"]), 1)
                        # nl_out = torch.stack((
                        #         encoders[args.contrastive.index("reports")]({
                        #             "input_ids": elem[task]["neg_input_ids"],
                        #             "attention_mask": elem[task]["neg_attention_mask"]
                        #         }),
                        #         encoders[args.contrastive.index("reports")]({
                        #             "input_ids": elem[task]["pos_input_ids"],
                        #             "attention_mask": elem[task]["pos_attention_mask"]
                        #         })
                        # ), dim=2)

                        # ft_out = torch.nn.functional.normalize(ft_out, dim=2)
                        # nl_out = torch.nn.functional.normalize(nl_out, dim=1)

                        # dist = torch.squeeze(torch.bmm(ft_out, nl_out), dim=1)
                        # dist = 1 / (1 + torch.exp(dist[:, 0] - dist[:, 1])) # softmax
                        # #predict = predict + dist.detach().tolist()
                        # temp_df[add_it] = dist.detach().numpy()
                        # add_it = add_it + 1
                        #temp_df[task[:-3]] = elem[task]
                        #temp_nd[:, add_it] = elem[task[:-3]].cpu().numpy()
                        
                        #print(task)
                        #print(encoders[args.contrastive.index("reports")])
                        #ok_rows = torch.where(elem[task]["neg_input_ids"][:, 0] != -1)[0].cpu().numpy()
                        #a1 = encoders[args.contrastive.index("reports")]({
                        #                "input_ids": elem[task]["neg_input_ids"][ok_rows],
                        #                "attention_mask": elem[task]["neg_attention_mask"][ok_rows]
                        #            })
                        #print("finished a1")
                        #print(encoders[args.contrastive.index("reports")])
                        #final_a1 = np.full((len(elem["id"][0]), a1.size()[1]), -1.0)
                        #final_a1[ok_rows] = a1.cpu().numpy()
                        #a2 = encoders[args.contrastive.index("reports")]({
                        #                "input_ids": elem[task]["pos_input_ids"][ok_rows],
                        #                "attention_mask": elem[task]["pos_attention_mask"][ok_rows]
                        #            })
                        #print("finished a2")
                        #final_a2 = np.full((len(elem["id"][0]), a2.size()[1]), -1.0)
                        #final_a2[ok_rows] = a2.cpu().numpy()
                        #nl_out = np.concatenate((final_a1, final_a2), axis=1)
                        # nl_out = torch.cat((
                        #             encoders[args.contrastive.index("reports")]({
                        #                 "input_ids": elem[task]["neg_input_ids"],
                        #                 "attention_mask": elem[task]["neg_attention_mask"]
                        #             }),
                        #             encoders[args.contrastive.index("reports")]({
                        #                 "input_ids": elem[task]["pos_input_ids"],
                        #                 "attention_mask": elem[task]["pos_attention_mask"]
                        #             })
                        #     ), dim=0)
                        #temp_nd[:, add_it + 1:add_it + 1 + 2 * args.repr_dim] = nl_out

                        #add_it = add_it + 1 + 2 * args.repr_dim

                #print(temp_nd, "temp nd")
                #print(df, "df pre")
                df = df.append(pd.DataFrame(
                    data=temp_nd,
                    index=elem["id"][0],
                    columns=columns
                ))
                #print(df, "df post")

    print(df.head(10))
    print(df.tail(10))

    df.to_csv(args.path_dir + "/outputs.csv")
    return 

def demo_basic(rank, world_size, args):
    torch.cuda.set_device(rank)
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    print(f"{rank} going to sleep...")
    time.sleep(rank * 10)
    print(f"{rank} waking up...")

    global datahandler

    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    sampler = torch.utils.data.distributed.DistributedSampler(datahandler.pretrain, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=datahandler.pretrain,
        shuffle=False,
        batch_size=args.bsz,
        sampler=sampler,
        **dataloader_kwargs
    )
    if verbose:
        pwint("[Pretraining] Completed data loading")

    ddp_models = []
    encoders = model.TCGAEncoders(data_types=args.contrastive, datahandler=datahandler, mode=args.mode, rep_dim=args.repr_dim, gpu=rank)
    for encoder in encoders:
        print(rank, "rank of encoder")
        torch.cuda.set_device(rank)
        encoder.cuda(rank)
        ddp_models.append(DDP(encoder, device_ids=[rank], output_device=rank))
    encoders = ddp_models
    #for it, encoder in enumerate(encoders):
    #    if rank == 0:
    #        ddp_models.append(DDP(encoders[it], device_ids=[rank]))
    #    else:
    #        encoder_copy = deepcopy(encoders[it])
    #        torch.cuda.set_device(rank)
    #        encoder_copy.cuda(rank)
    #        ddp_models.append(DDP(encoder_copy, device_ids=[rank]))
    #encoders = ddp_models

    # optimization
    optimizers = []
    for encoder in encoders:
        if isinstance(encoder, model.TransformerWithMLP):
            trns_optimizer = torch.optim.AdamW(encoder.module.trns.parameters(), lr=args.l_lr[0], weight_decay=args.wd)
            mlp_optimizer = torch.optim.AdamW(encoder.module.mlp.parameters(), lr=args.l_lr[1], weight_decay=args.wd)
            
            optimizers += [mlp_optimizer]
        elif isinstance(encoder, model.SimpleMLP):
            mlp_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.g_lr, weight_decay=args.wd)

            optimizers += [mlp_optimizer] # TODO: use the LRscheduler
        elif isinstance(encoder, CombTabTransformer):
            trns_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.c_lr, weight_decay=args.wd)

            optimizers += [trns_optimizer]

    if verbose:
        pwint("[Pretraining] Model generation complete, training begins")

    # logging
    start = time.time()
    # file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w') 
    if rank == 0:
        os.makedirs(args.path_dir, exist_ok=True)
        for encoder, dtype in zip(encoders, args.contrastive):
            #if len(args.gpu) > 1:
            #    torch.save(dict(epoch=0, state_dict=encoder.module.state_dict()), os.path.join(args.path_dir, "0-" + name + ".pth"))
            #else: # TODO: note that this will force it to be loaded with nn.DataParallel again
            torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, "0-" + dtype + ".pth"))

        data_args = vars(args)

        with open(os.path.join(args.path_dir, '0.args'), 'w') as fp:
            json.dump(data_args, fp, indent=4)

    saved_loss = [-1]
    saved_vars = []
    for x in args.contrastive:
        saved_vars.append([-1])

    distance = "cosine"
    if args.euclidean:
        distance = "euclidean"

    for e in range(1, args.epochs + 1):
        sampler.set_epoch(e)
        if e % args.progress_every == 0:
            with torch.no_grad():
                pwint("epoch ", e, "!")
                pwint("loss: ", torch.mean(torch.Tensor(saved_loss)))
                for it, dtype in enumerate(args.contrastive):
                    pwint(f"{dtype} var:", torch.mean(torch.Tensor(saved_vars[it])))

        for encoder in encoders:
            encoder.train()

        saved_loss = []
        for var in saved_vars:
            var = []

        for it, elem in tenumerate(train_loader):
            for encoder in encoders:
                encoder.zero_grad()

            outs = []
            for it, dtype in enumerate(args.contrastive):
                outs.append(encoders[it](datahandler.mask(elem[dtype], ratio=args.mask))) # TODO: remove the final layer when evaluating (ie make the last layer a projector)
            l = info_nce(outs, distance=distance)

            saved_loss.append(l.detach().item())
            for it, out in enumerate(outs):
                saved_vars[it].append(torch.mean(torch.std(out, dim=0)).detach())

            l.backward()
            for encoder in encoders:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
            for optimizer in optimizers:
                optimizer.step()
            torch.cuda.empty_cache()

        if rank == 0 and e % args.val_every == 0:
            #evaluate_single(args, [encoder.module for encoder in encoders], datahandler, e, tasks=[])
            evaluate_single(args, encoders, datahandler, e, tasks=[])

        if rank == 0 and e % args.eval_every == 0:
            #evaluate_single(args, [encoder.module for encoder in encoders], datahandler, e, expensive=True)
            evaluate_single(args, encoders, datahandler, e, expensive=True)

        if rank == 0 and e % args.save_every == 0:
            for encoder, name in zip(encoders, args.contrastive):
                torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, f'{e}-{name}.pth'))

            with open(os.path.join(args.path_dir, f'{e}.args'), 'w') as fp:
                json.dump(data_args, fp, indent=4)
            pwint("[saved]")

    if verbose:
        pwint("[Pretraining] Training complete")

    cleanup()

def pretrain(args, encoders, datahandler):
    # world_size=4
    # mp.spawn(demo_basic,
    #          args=(world_size,args),
    #          nprocs=world_size,
    #          join=True)

    # for it, encoder in enumerate(encoders):
    #     checkpoint = torch.load(os.path.join(args.path_dir, f"{args.num_epochs}-{args.contrastive[it]}.pth"))
    #     torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint['model_dict'],prefix='module.')
    #     encoder.load_state_dict(checkpoint)

    # return encoders

    # dataset
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    if args.site_batch != 1:
        sampler = datasets.SiteSampler(datahandler.pretrain, num_matches=args.site_batch)
        train_loader = torch.utils.data.DataLoader(
            dataset=datahandler.pretrain,
            batch_size=args.bsz,
            sampler=sampler,
            **dataloader_kwargs
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=datahandler.pretrain,
            shuffle=True,
            batch_size=args.bsz,
            **dataloader_kwargs
        )

    if verbose:
        pwint("[Pretraining] Completed data loading")

    # optimization
    optimizers = []
    lr_schedulers = []
    for encoder in encoders:
        if isinstance(encoder, model.TransformerWithMLP):
            trns_optimizer = torch.optim.AdamW(encoder.trns.parameters(), lr=args.l_lr[0], weight_decay=args.wd)
            mlp_optimizer = torch.optim.AdamW(encoder.mlp.parameters(), lr=args.l_lr[1], weight_decay=args.wd)
            
            optimizers += [trns_optimizer, mlp_optimizer]
            
            trns_lr_scheduler = LRScheduler(
                    optimizer=trns_optimizer,
                    warmup_epochs=args.warmup_epochs,
                    warmup_lr=0 if args.flat_lr or args.cosine_lr else args.l_lr[0] * args.bsz / 256,
                    num_epochs=args.epochs,
                    base_lr=args.l_lr[0] * args.bsz / 256,
                    final_lr=0 if args.cosine_lr else args.l_lr[0] * args.bsz / 256,
                    iter_per_epoch=len(train_loader),
                    constant_predictor_lr=True,
                    flat_warmup=args.flat_lr
            )
            mlp_lr_scheduler = LRScheduler(
                    optimizer=mlp_optimizer,
                    warmup_epochs=args.warmup_epochs,
                    warmup_lr=0 if args.cosine_lr else args.l_lr[1] * args.bsz / 256,
                    num_epochs=args.epochs,
                    base_lr=args.l_lr[1] * args.bsz / 256,
                    final_lr=0 if args.cosine_lr else args.l_lr[1] * args.bsz / 256,
                    iter_per_epoch=len(train_loader),
                    constant_predictor_lr=True,
                    flat_warmup=False
            )

            # lr_schedulers += [trns_lr_scheduler, mlp_lr_scheduler]
        elif isinstance(encoder, model.SimpleMLP):
            mlp_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.g_lr, weight_decay=args.wd)

            optimizers += [mlp_optimizer] # TODO: use the LRscheduler

            lr_scheduler = LRScheduler(
                    optimizer=mlp_optimizer,
                    warmup_epochs=args.warmup_epochs,
                    warmup_lr=0 if args.cosine_lr else args.g_lr * args.bsz / 256,
                    num_epochs=args.epochs,
                    base_lr=args.g_lr * args.bsz / 256,
                    final_lr=0 if args.cosine_lr else args.g_lr * args.bsz / 256,
                    iter_per_epoch=len(train_loader),
                    constant_predictor_lr=True,
                    flat_warmup=False
            )

            lr_schedulers += [lr_scheduler]
        elif isinstance(encoder, CombTabTransformer):
            trns_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.c_lr, weight_decay=args.wd)

            optimizers += [trns_optimizer]

            lr_scheduler = LRScheduler(
                    optimizer=trns_optimizer,
                    warmup_epochs=args.warmup_epochs,
                    warmup_lr=0 if args.cosine_lr else args.c_lr * args.bsz / 256,
                    num_epochs=args.epochs,
                    base_lr=args.c_lr * args.bsz / 256,
                    final_lr=0 if args.cosine_lr else args.c_lr * args.bsz / 256,
                    iter_per_epoch=len(train_loader),
                    constant_predictor_lr=True,
                    flat_warmup=False
            )

            lr_schedulers += [lr_scheduler]
    # dataparallel (multi gpu) TODO: use DistributedDataParallel for speedup
    if len(args.gpu) > 1:
        multi_encoders = []
        for encoder in encoders:
            #int_gpu = [int(x) for x in string_to_list(args.gpu)]
            #print(int_gpu, "int gpu")
            #multi_encoders.append(nn.DataParallel(encoder, device_ids=int_gpu))
            #multi_encoders.append(convert_model(nn.DataParallel(encoder.to(args.gpu[0]), device_ids=args.gpu)).cuda())
            multi_encoders.append(convert_model(nn.DataParallel(encoder)).cuda())
        encoders = multi_encoders


    if verbose:
        pwint("[Pretraining] Model generation complete, training begins")

    # logging
    start = time.time()
    # file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w') 
    os.makedirs(args.path_dir, exist_ok=True)
    for encoder, dtype in zip(encoders, args.contrastive):
        #if len(args.gpu) > 1:
        #    torch.save(dict(epoch=0, state_dict=encoder.module.state_dict()), os.path.join(args.path_dir, "0-" + name + ".pth"))
        #else: # TODO: note that this will force it to be loaded with nn.DataParallel again
        torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, "0-" + dtype + ".pth"))

    data_args = vars(args)

    with open(os.path.join(args.path_dir, '0.args'), 'w') as fp:
        json.dump(data_args, fp, indent=4)

    saved_loss = [-1]
    saved_vars = []
    for x in args.contrastive:
        saved_vars.append([-1])

    distance = "cosine"
    if args.euclidean:
        distance = "euclidean"

    for e in range(1, args.epochs + 1):
        if e % args.progress_every == 0:
            with torch.no_grad():
                pwint("epoch ", e, "!")
                pwint("loss: ", torch.mean(torch.Tensor(saved_loss)))
                for it, dtype in enumerate(args.contrastive):
                    pwint(f"{dtype} var:", torch.mean(torch.Tensor(saved_vars[it])))

                # file_to_update.write(f"epoch {e}" + '\n')
                # file_to_update.write(f"loss {torch.mean(torch.Tensor(saved_loss))}" + '\n')
                #for it, dtype in enumerate(args.contrastive):
                #    file_to_update.write(f"{dtype} var: {torch.mean(torch.Tensor(saved_vars[it]))} \n")
                # file_to_update.flush()
        
        for encoder in encoders:
            encoder.train()

        saved_loss = []
        for var in saved_vars:
            var = []

        #train_loader = datahandler.site_loader('pretrain', args.bsz)
        train_tloader = datahandler.by_type(args.type_bsz, select_size=args.type_num, dataset='pretrain')
        for it, elem in tenumerate(chain(train_loader, train_tloader), total=len(train_loader) + args.type_num):
            for encoder in encoders:
                encoder.zero_grad()

            outs = []
            if args.mixup_scale > 0:
                perm_idx, lam, elem = datahandler.mixup(elem, scale=args.mixup_scale)
            for it, dtype in enumerate(args.contrastive):
                outs.append(encoders[it](elem[dtype]))
            if args.mixup_scale > 0:
                l = info_nce(outs, distance=distance, twoway=True, lam=lam, perm_idx=perm_idx, temperature=args.temp)
            else:
                l = info_nce(outs, distance=distance, twoway=args.twoway, fourway=args.fourway, lr_weight=args.lr_weight, temperature=args.temp)

            saved_loss.append(l.detach().item())
            for it, out in enumerate(outs):
                saved_vars[it].append(torch.mean(torch.std(out, dim=0)).detach())

            l.backward()
            for encoder in encoders:
                if args.clip != -1:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
            for optimizer in optimizers:
                optimizer.step()
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
            torch.cuda.empty_cache()

        if e % args.val_every == 0:
            evaluate_single(args, encoders, datahandler, e, tasks=[])

        if e % args.eval_every == 0:
            evaluate_single(args, encoders, datahandler, e, expensive=True)

        if e % args.save_every == 0:
            for encoder, name in zip(encoders, args.contrastive):
                torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, f'{e}-{name}.pth'))

            with open(os.path.join(args.path_dir, f'{e}.args'), 'w') as fp:
                json.dump(data_args, fp, indent=4)
            pwint("[saved]")

    write_all(args, encoders, datahandler)

    if verbose:
        pwint("[Pretraining] Training complete")

    return encoders

def evaluate(args, encoders, datahandler):
    try:
        encoders.module
    except:
        encoders = [torch.nn.DataParallel(encoder) for encoder in encoders]
    ft_encoders = evaluate_single(args, encoders, datahandler, -1, expensive=True)
    return ft_encoders

def main(args):
    global verbose, log_file

    if args.verbose:
        verbose = True

    if args.path_dir == "":
        raise UserWarning("please do not pass empty experiment names")
    args.path_dir = '../save/' + args.path_dir
    try:
        os.mkdir(args.path_dir)
    except:
        pass
    log_file = args.path_dir + '/all.log'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # TODO: add all GPUs to make parallelism work 
    pwint("gpu", args.gpu)
    pwint(torch.cuda.device_count())

    args.l_lr = [float(x) for x in string_to_list(args.l_lr)]
    args.ft_lr = [float(x) for x in string_to_list(args.ft_lr)]
    args.manual = string_to_list(args.manual)
    args.rna_hidden = [int(x) for x in string_to_list(args.rna_hidden)]
    args.clin_hidden = [int(x) for x in string_to_list(args.clin_hidden)]
    args.lr_weight = [float(x) for x in string_to_list(args.lr_weight)]

    if "validate" in args.mode:
        if args.mode == "validate":
            pass
        else:
            raise ValueError("validation mode is exclusive. do not add other modes, input 'validate' exactly")
    else:
        args.mode = string_to_list(args.mode)

    if args.eval_epoch == -1:
        try:
            args.eval_epoch = most_recent_file(args.path_dir, "args")
            args.eval_epoch = int(args.eval_epoch[args.eval_epoch.rfind("/") + 1:args.eval_epoch.rfind(".")])
        except:
            args.eval_epoch = -1

    has_saved = os.path.isfile(os.path.join(args.path_dir, f"{args.eval_epoch}.args")) 
    print(has_saved, args.eval_epoch)
    new_weights = args.new_weights or (not has_saved)
    print(new_weights, "new weights")

    if new_weights:
        if args.contrastive == "":
            raise ValueError("contrastive data-type cannot be empty. specify contrastive argument")

        args.contrastive = string_to_list(args.contrastive)
        args.zero_shot = string_to_list(args.zero_shot)
        args.finetune = string_to_list(args.finetune)

        if ("pretrain" in args.mode or "evaluate" in args.mode) and len(args.contrastive) < 2:
            raise NotImplementedError("unimodal contrastive not yet implemented. add at least 2 contrastive data types")

        if "validate" in args.mode and len(args.zero_shot + args.finetune) == 1:
            raise ValueError("you must specify a target for validation modes")
    else:
        if args.contrastive != "" or args.zero_shot != "" or args.finetune != "":
            raise ValueError("do not specify data types when loading from existing exp")
        if args.train_ratio != 0.8 or args.ft_train_ratio != 0.5:
            raise ValueError("cannot specify split when loading a pre-existing dataset")


    if "validate" in args.mode:
        datahandler = datasets.TCGADataHandler(contrastive=args.contrastive, zero_shot=args.zero_shot, finetune=args.finetune, train_ratio=args.train_ratio, ft_train_ratio=args.ft_train_ratio, lg_types=args.lg_types)
        encoder = model.TCGAEncoders(data_types=args.contrastive, datahandler=datahandler, mode=args.mode, rep_dim=args.repr_dim)[0]
    else:
        if not new_weights: 
            #datahandler = torch.load(os.path.join(args.path_dir, "dataset/datahandler.pt"))
            with open(os.path.join(args.path_dir, 'dataset/datahandler.pt'), 'rb') as f:
                datahandler = pickle.load(f)
            
            with open(os.path.join(args.path_dir, str(args.eval_epoch) + ".args"), "r") as f:
                old_args = json.load(f)

            args.contrastive = datahandler.contrastive
            args.zero_shot = datahandler.zero_shot
            args.finetune = datahandler.finetune

            #encoders = model.TCGAEncoders(data_types=args.contrastive, datahandler=datahandler, mode=args.mode, rep_dim=old_args["repr_dim"])
            encoders = model.TCGAEncoders(data_types=old_args["contrastive"], datahandler=datahandler, mode=args.mode, rep_dim=old_args["repr_dim"], rna_hidden=old_args["rna_hidden"], trns_arch=old_args["lm_arch"], clin_arch=old_args["clin_arch"], clin_hidden=old_args["clin_hidden"], cheads=old_args["cheads"], cdepth=old_args["cdepth"], cdropout=old_args["cdropout"], nocombine=old_args["nocombine"], inter_attn=old_args["inter_attn"], cdims=old_args["cdims"])
            for encoder, name in zip(encoders, args.contrastive): # TODO REMOVE THIS
                try:
                    encoder.load_state_dict(torch.load(os.path.join(args.path_dir, str(args.eval_epoch) + "-" + name + ".pth"))["state_dict"])
                except:
                    sdict = torch.load(os.path.join(args.path_dir, str(args.eval_epoch) + "-" + name + ".pth"))["state_dict"]
                    for key in list(sdict.keys()):
                        if key.startswith("module."):
                            sdict[key[7:]] = sdict[key]
                            del sdict[key]
                    encoder.load_state_dict(sdict)
        else:
            datahandler = datasets.TCGADataHandler(contrastive=args.contrastive, zero_shot=args.zero_shot, finetune=args.finetune, train_ratio=args.train_ratio, ft_train_ratio=args.ft_train_ratio, lg_types=args.lg_types, rna_thresh=args.rna_thresh, clin_thresh=args.clin_thresh, rna_set=args.rna_set, rand_shuffle=args.rand_shuffle, lm_arch=args.lm_arch, clin_one_hot=(args.clin_arch == 'mlp'))
            encoders = model.TCGAEncoders(data_types=args.contrastive, datahandler=datahandler, mode=args.mode, rep_dim=args.repr_dim, rna_hidden=args.rna_hidden, trns_arch=args.lm_arch, clin_arch=args.clin_arch, clin_hidden=args.clin_hidden, cheads=args.cheads, cdepth=args.cdepth, cdropout=args.cdropout, nocombine=args.nocombine, inter_attn=args.inter_attn, cdims=args.cdims)
            print(encoders)
            
            try:
                os.mkdir(os.path.join(args.path_dir, "dataset"))
            except:
                pass

            torch.save(datahandler, os.path.join(args.path_dir, "dataset/datahandler.pt"))
            with open(os.path.join(args.path_dir, "dataset/datahandler.pt"), 'wb') as f:
                pickle.dump(datahandler, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    try:
        if "validate" in args.mode:
            encoder = validate(args, encoder, datahandler)
        if "pretraining" in args.mode:
            encoders = pretrain(args, encoders, datahandler)
        if "evaluate" in args.mode:
            encoders = evaluate(args, encoders, datahandler)
    except Exception:
        pwint(traceback.format_exc())

    if not args.silent:
        for i in range(0,15):
            subprocess.run("echo $\'\a\'", shell=True)
            time.sleep(3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    """
        MODE
        runs MULTIPLE modes all at once! with (), nospace syntax. finetune
        and inverse are mutually exclusive.  validate is mutually exclusive
        with all other modes.

        choices:
        validate: validate whether a given architecture is working. the
            architecture to be validated is determined by tcga_src[0], and the
            target is tcga_src[2]. no other modes should be indicated.
        pretraining: complete pretraining with tcga_src.
        finetune: finetune on a given target, indicated by tcga_src[2].
        inverse: inverse classification, a la CLIP "zero shot ResNet"
    """
    parser.add_argument('--mode', default='(validate,pretraining,evaluate)', type=str)

    """
        GPU
        which gpu is used 
    """
    parser.add_argument('--gpu', default="0,1,2,3,4,5,6,7", type=str)

    # Data generation options
    parser.add_argument('--contrastive', default='', type=str) # tcga data type for left (primary) encoder
    parser.add_argument('--zero_shot', default='', type=str) # tcga data type for right (secondary) encoder
    parser.add_argument('--finetune', default='', type=str) # target for validation/evaluation task

    parser.add_argument('--train_ratio', default=0.8, type=float) # pretrain/train/test split
    parser.add_argument('--ft_train_ratio', default=0.5, type=float) # pretrain/train/test split

    # File I/O
    """
        PATH_DIR
        directory in which to save the results from this experiment
    """
    parser.add_argument('--path_dir', default='', type=str)
    parser.add_argument('--new_weights', default=False, action='store_true') # whether to use new weights/data
    parser.add_argument('--lg_types', default=False, action='store_true')

    """
        EVAL_EPOCH
        which epoch to use for the eval/inverse loops

        -1 defaults to the last epoch; you can choose others or pick a range
        (the latter is not implemented yet)
    """
    parser.add_argument('--eval_epoch', default=-1, type=int)

    parser.add_argument('--rna_thresh', default=0.5, type=float)
    parser.add_argument('--rna_set', default='', type=str)
    parser.add_argument('--clin_thresh', default=50, type=int)

    parser.add_argument('--rna_hidden', default='1280', type=str)
    parser.add_argument('--lm_arch', default='distilbert', type=str)
    parser.add_argument('--clin_arch', default='tabtransformer', type=str)
    parser.add_argument('--clin_hidden', default='[]', type=str)
    parser.add_argument('--cdepth', default=1, type=int)
    parser.add_argument('--cheads', default=6, type=int)
    parser.add_argument('--nocombine', default=False, action='store_true')
    parser.add_argument('--inter_attn', default=False, action='store_true')
    parser.add_argument('--cdropout', default=0.1, type=float)
    parser.add_argument('--cdims', default=32, type=int)

    # Training reporting options
    parser.add_argument('--progress_every', default=1, type=int)
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--val_every', default=1, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--silent', default=False, action='store_true')
    parser.add_argument('--print_results', default=False, action='store_true')
    parser.add_argument('--compare_nl', default=False, action='store_true')

    # Optimizer options
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--ft_epochs', default=20, type=int)
    parser.add_argument('--ft_trials', default=20, type=int)
    #parser.add_argument('--freeze_model', default=True, action='store_false')

    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--val_bsz', default=16, type=int)
    parser.add_argument('--manual', default='(brca)', type=str)
    parser.add_argument('--type_bsz', default=32, type=int)
    parser.add_argument('--type_num', default=0, type=int)
    parser.add_argument('--type_reps', default=3, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)

    parser.add_argument('--rand_shuffle', default=False, action='store_true')
    parser.add_argument('--site_batch', default=1, type=int)

    parser.add_argument('--l_lr', default="(2e-5,0.001)", type=str) # learning rate for bert
    parser.add_argument('--g_lr', default=0.001, type=float)
    parser.add_argument('--c_lr', default=1e-4, type=float)
    parser.add_argument('--ft_lr', default="(2e-5,0.001)", type=str) # learning rate for fine tuning
    parser.add_argument('--wd', default=0.001, type=float) # TODO: add different weight decays for different architectures
    #parser.add_argument('--p', default=0.9, type=float) # momentum for genetic mlp
    parser.add_argument('--euclidean', default=False, action='store_true')
    parser.add_argument('--twoway', default=False, action='store_true')
    parser.add_argument('--lr_weight', default='(1,1)', type=str)
    parser.add_argument('--fourway', default=False, action='store_true')
    parser.add_argument('--cosine_lr', default=False, action='store_true')
    parser.add_argument('--flat_lr', default=False, action='store_true')

    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--clip', default=-1.0, type=float)
    parser.add_argument('--mask', default=0.0, type=float)
    parser.add_argument('--mixup_scale', default=0.0, type=float)

    # NN size options
    parser.add_argument('--repr_dim', default=32, type=int)

    args = parser.parse_args()
    main(args)
