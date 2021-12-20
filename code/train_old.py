# python imports
import os
import gc
import subprocess
import traceback
import argparse
import datetime
from tqdm.contrib import tenumerate
import time
import random
from PIL import Image
import pickle
import json
from copy import deepcopy
from itertools import chain
from sklearn.linear_model import LogisticRegression

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

# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchmetrics

# local imports
import datasets_mid as datasets # TODO: make sure this import actually works
import model
from tabtransformer.tabtransformer.tab_transformer_pytorch import CombTabTransformer
from sync_batchnorm.sync_batchnorm import convert_model

verbose = False

nl_target = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

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


class LRScheduler(object):
    """
    Learning rate scheduler for the optimizer.

    Warmup increases to base linearly, while base decays to final using cosine.
    """

    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
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

def _uni_info_nce(z1, z2, temperature=0.1, distance="cosine", both_sides=True, remove_duplicates=False, targets=None):
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

    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def info_nce(z, temperature=0.1, distance="cosine", both_sides=True):
    # wrapper to do infonce on multiple contrastive objectives
    loss = []
    for it1, z1 in enumerate(z):
        for it2, z2 in enumerate(z):
            if it1 == it2:
                continue
            
            loss.append(torch.unsqueeze(_uni_info_nce(z1, z2, temperature, distance, both_sides), dim=0))
    loss = torch.mean(torch.cat(loss))

    return loss

def confusion_matrix_str(cm, normalize=True, figs=3, label_names=None):
    if not normalize:
        raise NotImplementedError("non normalized arrays not available yet")
    else: 
        cm = cm / np.sum(cm, axis=0, keepdims=True)

    n = np.size(cm, axis=0)
    color_scale = ["\u001b[38;5;$17m", "\u001b[38;5;$60m", "\u001b[38;5;$109m", "\u001b[38;5;$137m", "\u001b[38;5;$167m", "\u001b[38;5;$196m"]
    bold = "\033[1m"
    reset = "\u001b[0m"
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
    file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w') 

    if tasks == [-1]:
        tasks = args.zero_shot + args.finetune

    if clip_inv:
        if verbose:
            print("[Evaluate] begin CLIP inverse evaluation...")

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
            if expensive:
                pcs_acc = []
                pcs_total = []
                pcs_manual = []
                pcs_mtotal = []

            distance = "cosine"
            if args.euclidean:
                distance = "euclidean"

            for it, elem in tenumerate(val_loader):
                with torch.no_grad():
                    base_out = encoders[encoder_idx](elem[args.contrastive[encoder_idx]])
                    comp_out = encoders[oit](elem[dtype])

                    acc.append(clip_acc(base_out, comp_out, distance=distance))
                    loss.append(info_nce([base_out, comp_out], distance=distance).item())
                    _, _, cm = clip_acc(base_out, comp_out, distance=distance, as_confusion_matrix=True, labels=elem["type"], label_size=datahandler.num_types)
                cms.append(cm)

            if expensive:
                type_loader = datahandler.by_type(args.type_bsz, select_size=-1, dataset='test', reps=args.type_reps) 
                for it, elem in tenumerate(type_loader):
                    with torch.no_grad():
                        base_out = encoders[encoder_idx](elem[args.contrastive[encoder_idx]])
                        comp_out = encoders[oit](elem[dtype])

                        if dtype == "reports":
                            with open("compare_nl.txt", "a") as f:
                                y_true = elem["reports"]["input_ids"]
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
                    pcs_acc.append(pc1)
                    pcs_total.append(pc2)

                if dtype == "reports":
                    for ctype in args.manual:
                        manual_loader = datahandler.manual_loader(args.type_bsz, reps=args.type_reps, type=ctype) 
                        
                        for it, elem in tenumerate(manual_loader):
                            with torch.no_grad():
                                base_out = encoders[encoder_idx](elem[args.contrastive[encoder_idx]])
                                comp_out = encoders[oit](elem["manual-" + dtype])

                                pc1, pc2, _ = clip_acc(base_out, comp_out, distance=distance, as_confusion_matrix=True, labels=elem["type"], label_size=datahandler.num_types)
                            pcs_manual.append(pc1)
                            pcs_mtotal.append(pc2)

            cms = np.sum(np.array(cms), axis=0)
            loss = np.array(loss)
            if expensive:
                pcs_acc = np.nanmean(np.array(pcs_acc), axis=0)
                pcs_sample = np.sum(np.array(pcs_total), axis=0) / args.type_reps
                pcs_manual = np.nanmean(np.array(pcs_manual), axis=0)
                pcs_msample = np.sum(np.array(pcs_mtotal), axis=0) / args.type_reps

            print(f"Data type {dtype}")
            print("Val acc:", np.around(acc, 3), np.around(np.mean(acc), 3))
            print("Val loss:", np.around(loss, 3), np.around(np.mean(loss), 3))
            print("Confusion Matrix:", confusion_matrix_str(cms, label_names=nl_target), sep='\n')
            if expensive:
                print("Indices | Within-Class Acc. | Mean Class Size | Model Class Size")
                print("".join('{:>6}'.format(nl_target[i]) for i in range(0, datahandler.num_types))[1:])
                print("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_acc * 100, 1)))[1:])
                print("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_sample, 1)))[1:])
                print("".join('{:>6}'.format(str(x)) for x in list(np.around(1 / pcs_acc, 1)))[1:])
                print(pcs_manual, "manual")
                print(pcs_msample, "sample")

            file_to_update.write(f"Data type {dtype}")
            file_to_update.write(f"Val acc: {np.around(acc, 3)} | {np.around(np.mean(acc), 3)}")
            file_to_update.write(f"Val loss: {np.around(loss, 3)} | {np.around(np.mean(loss), 3)}")
            file_to_update.write(f"Confusion matrix: \n {confusion_matrix_str(cms)}")
            if expensive:
                file_to_update.write("Indices | Within-Class Acc. | Mean Class Size | Model Class Size")
                file_to_update.write("".join('{:>6}'.format(nl_target[i]) for i in range(0, datahandler.num_types))[1:])
                file_to_update.write("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_acc * 100, 1)))[1:])
                file_to_update.write("".join('{:>6}'.format(str(x)) for x in list(np.around(pcs_sample, 1)))[1:])
                file_to_update.write("".join('{:>6}'.format(str(x)) for x in list(np.around(1 / pcs_acc, 1)))[1:])
                file_to_update.write(f"{pcs_manual} manual")
                file_to_update.write(f"{pcs_msample} sample")
            file_to_update.flush()

    ft_encoders = []
    if expensive and len(tasks) > 0:
        print(f"[Evaluate] begin evaluation on tasks {tasks}")

        for it, task in enumerate(tasks):
            #if finetune:
            print(f"[Evaluate] begin task {task} evaluation ({it} out of {len(tasks)})")

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
                output_nclasses = int(max(torch.max(datahandler.val_train[task][:][task]).item(), torch.max(datahandler.val_test[task][:][task]).item()) + 1)

                ft_encoder = model.WithFinetuneLayers(encoders[encoder_idx], args.repr_dim, output_size=output_nclasses, size=[])
                ft_encoders.append(ft_encoder)

                if finetune:
                    model_optimizer = torch.optim.AdamW(ft_encoder.model.parameters(), lr=args.ft_lr[0], weight_decay=args.wd)
                ft_optimizer = torch.optim.AdamW(ft_encoder.finetune.parameters(), lr=args.ft_lr[1], weight_decay=args.wd)

                for e in range(1, 2 + args.ft_epochs):
                    # evaluate
                    ft_encoder.eval()

                    correct = 0
                    total = 0
                    saved_loss = []
                    with torch.no_grad():
                        for it, elem in enumerate(test_loader):
                            out = ft_encoder(elem[args.contrastive[encoder_idx]])

                            if output_nclasses != 2:
                                loss = torch.nn.CrossEntropyLoss()
                                l = loss(out, elem[task].long())
                            else:
                                loss = torch.nn.BCEWithLogitsLoss()
                                l = loss(out, torch.nn.functional.one_hot(elem[task].long(), num_classes=output_nclasses).float())
                            saved_loss.append(float(l.item()))

                            out = torch.argmax(out, dim=1)
                            correct += torch.numel(torch.where(out == elem[task])[0])
                            total += int(elem[list(elem.keys())[0]].size()[0])
                    print(f"[Evaluate] on epoch {e - 1}, task {task}: accuracy {round(correct/total, 3)}")
                    file_to_update.write(f"epoch {e - 1} & task {task} & zero shot: accuracy {round(correct/total, 3)}")
                    print(f"test loss: {np.mean(np.array(saved_loss))}")

                    if e > args.ft_epochs:
                        if save_num != -1:
                            torch.save(dict(epoch=0, state_dict=ft_encoder.state_dict()), os.path.join(args.path_dir, str(save_num) + "-" + task + ".pth"))
                        break

                    # train
                    ft_encoder.train()

                    print(f"[Evaluate] epoch: {e} out of {args.ft_epochs}")
                    saved_loss = []
                    for it, elem in enumerate(train_loader):
                        out = ft_encoder(elem[args.contrastive[encoder_idx]]) 

                        if output_nclasses != 2:
                            loss = torch.nn.CrossEntropyLoss()
                            l = loss(out, elem[task].long())
                        else:
                            loss = torch.nn.BCEWithLogitsLoss()
                            l = loss(out, torch.nn.functional.one_hot(elem[task].long(), num_classes=output_nclasses).float())
                        saved_loss.append(float(l.detach().item()))

                        l.backward()
                        if finetune:
                            model_optimizer.step()
                        ft_optimizer.step()
                    print(f"train loss: {np.mean(np.array(saved_loss))}")

            # else:
            #     ft_encoder = deepcopy(encoders[encoder_idx])
            #     if "reports" in args.contrastive:
            #         nl_encoder = encoders[args.contrastive.index("reports")]
            #     elif "clean-reports" in args.contrastive:
            #         nl_encoder = encoders[args.contrastive.index("clean-reports")]
            #     else:
            #         raise ValueError("nl target but can't find nl encoder")
            #     
            #     ft_optimizer = torch.optim.AdamW(ft_encoder.parameters(), lr=args.ft_lr[0], weight_decay=args.wd)
            #     nl_optimizer = torch.optim.AdamW(nl_encoder.parameters(), lr=args.ft_lr[0], weight_decay=args.wd)

            #     label_size = torch.numel(torch.unique(datahandler.val_train[task][:][task[:-3]]))

            #     for e in range(1, 2 + args.ft_epochs):
            #         # evaluate
            #         ft_encoder.eval()
            #         nl_encoder.eval()

            #         acc = []
            #         cm = []
            #         with torch.no_grad():
            #             for it, elem in enumerate(test_loader):
            #                 ft_out = ft_encoder(elem[args.contrastive[encoder_idx]])
            #                 nl_out = nl_encoder(elem["reports"])

            #                 #acc.append(clip_acc(ft_out, nl_out, distance=distance, remove_duplicates=True))
            #                 cmmm = clip_acc(ft_out, nl_out, distance=distance, as_confusion_matrix=True, labels=elem[task[:-3]], label_size=label_size)[2]
            #                 acc.append(cmmm[0, 0] * (1 - torch.sum(elem[task[:-3]]) / torch.numel(elem[task[:-3]])) + cmmm[1, 1] * torch.sum(elem[task[:-3]])/torch.numel(elem[task[:-3]]))
            #                 cm.append(cmmm) 
            #                 
            #         acc = np.mean(np.array(acc))
            #         cm = np.array(cm)
            #         if len(cm.shape) > 2:
            #             cm = np.mean(cm, axis=0)
            #         print(f"[Evaluate] on epoch {e - 1}, task {task}: accuracy {round(float(acc), 3)}")
            #         print(f"[Evaluate] cofusion matrix: {confusion_matrix_str(cm, label_names=[str(x) for x in range(label_size)])}")
            #         file_to_update.write(f"epoch {e - 1} & task {task} & zero shot: accuracy {round(float(acc), 3)}")

            #         if e > args.ft_epochs:
            #             if save_num != -1:
            #                 torch.save(dict(epoch=0, state_dict=ft_encoder.state_dict()), os.path.join(args.path_dir, str(save_num) + "-" + task + ".pth"))
            #             break

            #         # train
            #         ft_encoder.train()
            #         nl_encoder.train()

            #         print(f"[Evaluate] nl-epoch: {e} out of {args.ft_epochs}")
            #         for it, elem in enumerate(train_loader):
            #             ft_out = ft_encoder(elem[args.contrastive[encoder_idx]])
            #             nl_out = nl_encoder(elem[task])

            #             l = _uni_info_nce(ft_out, nl_out, distance=distance, remove_duplicates=True, both_sides=False)

            #             l.backward()
            #             if finetune:
            #                 model_optimizer.step()
            #             ft_optimizer.step()
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

def pretrain(args, encoders, datahandler):
    # dataset
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(
        dataset=datahandler.pretrain,
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )

    if verbose:
        print("[Pretraining] Completed data loading")

    # optimization
    optimizers = []
    for encoder in encoders:
        if isinstance(encoder, model.TransformerWithMLP):
            trns_optimizer = torch.optim.AdamW(encoder.trns.parameters(), lr=args.l_lr[0], weight_decay=args.wd)
            mlp_optimizer = torch.optim.AdamW(encoder.mlp.parameters(), lr=args.l_lr[1], weight_decay=args.wd)
            
            optimizers += [trns_optimizer, mlp_optimizer]
        elif isinstance(encoder, model.SimpleMLP):
            mlp_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.g_lr, weight_decay=args.wd)

            optimizers += [mlp_optimizer] # TODO: use the LRscheduler
        elif isinstance(encoder, CombTabTransformer):
            trns_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.c_lr, weight_decay=args.wd)

            optimizers += [trns_optimizer]
    # dataparallel (multi gpu) TODO: use DistributedDataParallel for speedup
    if len(args.gpu) > 1:
        multi_encoders = []
        for encoder in encoders:
            #int_gpu = [int(x) for x in string_to_list(args.gpu)]
            #print(int_gpu, "int gpu")
            #multi_encoders.append(nn.DataParallel(encoder, device_ids=int_gpu))
            multi_encoders.append(convert_model(nn.DataParallel(encoder)).cuda())
        encoders = multi_encoders

    # lr_scheduler = LRScheduler(
    #     optimizer=optimizer,
    #     warmup_epochs=args.warmup_epochs,
    #     warmup_lr=0,
    #     num_epochs=args.epochs,
    #     base_lr=args.lr * args.bsz / 256,
    #     final_lr=0,
    #     iter_per_epoch=len(train_loader),
    #     constant_predictor_lr=True
    # )

    if verbose:
        print("[Pretraining] Model generation complete, training begins")

    # logging
    start = time.time()
    file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w') 
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
                print("epoch ", e, "!")
                print("loss: ", torch.mean(torch.Tensor(saved_loss)))
                for it, dtype in enumerate(args.contrastive):
                    print(f"{dtype} var:", torch.mean(torch.Tensor(saved_vars[it])))

                file_to_update.write(f"epoch {e}" + '\n')
                file_to_update.write(f"loss {torch.mean(torch.Tensor(saved_loss))}" + '\n')
                for it, dtype in enumerate(args.contrastive):
                    file_to_update.write(f"{dtype} var: {torch.mean(torch.Tensor(saved_vars[it]))} \n")
                file_to_update.flush()
        
        for encoder in encoders:
            encoder.train()

        saved_loss = []
        for var in saved_vars:
            var = []

        train_tloader = datahandler.by_type(args.type_bsz, select_size=args.type_num, dataset='pretrain')
        for it, elem in tenumerate(chain(train_loader, train_tloader), total=len(train_loader) + args.type_num):
            for encoder in encoders:
                encoder.zero_grad()

            outs = []
            for it, dtype in enumerate(args.contrastive):
                outs.append(encoders[it](elem[dtype])) # TODO: remove the final layer when evaluating (ie make the last layer a projector)
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

        if e % args.val_every == 0:
            evaluate_single(args, encoders, datahandler, e, tasks=[])

        if e % args.eval_every == 0:
            evaluate_single(args, encoders, datahandler, e, expensive=True)

        if e % args.save_every == 0:
            for encoder, name in zip(encoders, args.contrastive):
                if len(args.gpu) > 1:
                    torch.save(dict(epoch=0, state_dict=encoder.module.state_dict()), os.path.join(args.path_dir, f'{e}-{name}.pth'))
                else:
                    torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, f'{e}-{name}.pth'))

            with open(os.path.join(args.path_dir, f'{e}.args'), 'w') as fp:
                json.dump(data_args, fp, indent=4)
            print("[saved]")

    if verbose:
        print("[Pretraining] Training complete")

    return encoders

def evaluate(args, encoders, datahandler):
    ft_encoders = evaluate_single(args, encoders, datahandler, 1, expensive=True)
    return ft_encoders

def main(args):
    global verbose

    if args.verbose:
        verbose = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # TODO: add all GPUs to make parallelism work 
    print("gpu", args.gpu)
    print(torch.cuda.device_count())

    if args.path_dir == "":
        raise UserWarning("please do not pass empty experiment names")
    args.path_dir = '../../save/' + args.path_dir

    args.l_lr = [float(x) for x in string_to_list(args.l_lr)]
    args.ft_lr = [float(x) for x in string_to_list(args.ft_lr)]
    args.manual = string_to_list(args.manual)

    if "validate" in args.mode:
        if args.mode == "validate":
            pass
        else:
            raise ValueError("validation mode is exclusive. do not add other modes, input 'validate' exactly")
    else:
        args.mode = string_to_list(args.mode)

    has_saved = os.path.isfile(os.path.join(args.path_dir, "0.pth")) 
    new_weights = args.new_weights or (not has_saved)

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
            datahandler = torch.load(os.path.join(args.path_dir, "dataset/datahandler.pt"))
            
            if args.eval_epoch == -1:
                args.eval_epoch = most_recent_file(args.path_dir, "pth")
                args.eval_epoch = args.eval_epoch[args.eval_epoch.rfind("/"):args.eval_epoch.rfind(".")]

            with open(os.path.join(args.path_dir, str(args.eval_epoch) + ".args"), "r") as f:
                old_args = json.load(f)

            args.contrastive = datahandler.contrastive
            args.zero_shot = datahandler.zero_shot
            args.finetune = datahandler.finetune

            encoders = model.TCGAEncoders(data_types=args.contrastive, datahandler=datahandler, mode=args.mode, rep_dim=old_args["repr_dim"])
            for encoder, name in zip(encoders, args.contrastive): # TODO REMOVE THIS
                encoder.load_state_dict(torch.load(os.path.join(args.path_dir, str(args.eval_epoch) + "-" + name + ".pth"))["state_dict"])
        else:
            datahandler = datasets.TCGADataHandler(contrastive=args.contrastive, zero_shot=args.zero_shot, finetune=args.finetune, train_ratio=args.train_ratio, ft_train_ratio=args.ft_train_ratio, lg_types=args.lg_types)
            encoders = model.TCGAEncoders(data_types=args.contrastive, datahandler=datahandler, mode=args.mode, rep_dim=args.repr_dim)
            
            try:
                os.mkdir(args.path_dir)
                os.mkdir(os.path.join(args.path_dir, "dataset"))
            except:
                pass

            torch.save(datahandler, os.path.join(args.path_dir, "dataset/datahandler.pt"))
    

    try:
        if "validate" in args.mode:
            encoder = validate(args, encoder, datahandler)
        if "pretraining" in args.mode:
            encoders = pretrain(args, encoders, datahandler)
        if "evaluate" in args.mode:
            encoders = evaluate(args, encoders, datahandler)
    except Exception:
        print(traceback.format_exc())

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

    # Training reporting options
    parser.add_argument('--progress_every', default=5, type=int)
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--val_every', default=10, type=int)
    parser.add_argument('--eval_every', default=50, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--silent', default=False, action='store_true')
    parser.add_argument('--print_results', default=False, action='store_true')

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
    #parser.add_argument('--warmup_epochs', default=5, type=int)

    parser.add_argument('--l_lr', default="(2e-5,0.001)", type=str) # learning rate for bert
    parser.add_argument('--g_lr', default=0.001, type=float)
    parser.add_argument('--c_lr', default=1e-4, type=float)
    parser.add_argument('--ft_lr', default="(2e-5,0.001)", type=str) # learning rate for fine tuning
    parser.add_argument('--wd', default=0.001, type=float) # TODO: add different weight decays for different architectures
    #parser.add_argument('--p', default=0.9, type=float) # momentum for genetic mlp
    parser.add_argument('--euclidean', default=False, action='store_true')

    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--clip', default=-1.0, type=float)

    # NN size options
    parser.add_argument('--repr_dim', default=32, type=int)

    args = parser.parse_args()
    main(args)

