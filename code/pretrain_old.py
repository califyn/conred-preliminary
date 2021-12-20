# python imports
import os
import subprocess
import argparse
import datetime
import time
import random
from PIL import Image
import pickle
import json
from copy import deepcopy

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

# local imports
import datasets # TODO: make sure this import actually works
import model

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

def string_to_list(in):
    in = in.replace('(', '["')
    in = in.replace(')', '"]')
    in = in.replace(',', '","')
    return eval(in)

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


def info_nce(z1, z2, temperature=0.1, distance="cosine", both_sides=False): # TODO: make sure this actually works
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
    if torch.cuda.is_available():
        logits = logits.cuda()

    n = z1.shape[0]
    if both_sides:
        labels = torch.arange(0, 2 * n, dtype=torch.long)
        labels = labels + (1 - labels % 2)
    else:
        labels = torch.arange(0, n, dtype=torch.long)

    if torch.cuda.is_available():
        labels = labels.cuda()

    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def validate(args, encoder, train_dataset, test_dataset):
    # dataset
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )

    if verbose:
        print("[Validation] Completed data loading")

    # optimization
    optimizer = torch.optim.SGD(
        encoder.parameters(),
        momentum=0.9,
        lr=args.lr,
        weight_decay=args.wd
    ) # extremely sensitive to learning rate
    lr_scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=0,
        num_epochs=args.epochs,
        base_lr=args.lr * args.bsz / 256,
        final_lr=0,
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True
    )

    if verbose:
        print("[Validation] Model generation complete, training begins")

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))

    data_args = vars(args)

    with open(os.path.join(args.path_dir, '0.args'), 'w') as fp:
        json.dump(data_args, fp, indent=4)

    for e in range(1, args.epochs + 1):
        encoder.train()

        for it, elem in enumerate(train_loader):
            encoder.zero_grad()

            loss = torch.nn.MSELoss(model(elem[0]), elem[test_output])

            loss.backward()
            optimizer.step()

        if e % args.progress_every == 0:
            with torch.no_grad():
                print("epoch ", e, "!")
                print("loss: ", loss.item())
                acc = 0
                total = 0
                for it, elem in enumerate(test_loader):
                    out = np.argmax(encoder(elem[0]).cpu().numpy())
                    if out == np.argmax(elem[2].cpu().numpy()):
                        acc = acc + 1
                    total = total + 1
                print("accuracy: ", acc / total)

        if e % args.save_every == 0:
            torch.save(dict(epoch=0, state_dict=encoder.state_dict()), os.path.join(args.path_dir, f'{e}.pth'))

            with open(os.path.join(args.path_dir, f'{e}.args'), 'w') as fp:
                json.dump(data_args, fp, indent=4)
            print("[saved]")

    if verbose:
        print("[Validation] Training complete")

    return encoder


def main(args):
    global verbose

    if args.verbose:
        verbose = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.path_dir == "":
        raise UserWarning("please do not pass empty experiment names")
    args.path_dir = '../../save/' + args.path_dir

    if "validate" in args.mode:
        if args.mode == "validate":
            pass
        else:
            raise ValueError("validation mode is exclusive. do not add other modes, input 'validate' exactly")
    else:
        args.mode = string_to_list(args.mode)

    if "finetune" in args.mode and "inverse" in args.mode:
        raise ValueError("either finetune or inverse mode, not both")

    if args.left == "":
        raise ValueError("left data-type cannot be empty. specify left argument")

    if ("pretrain" in args.mode or "inverse" in args.mode) and args.right == "":
        raise NotImplementedError("unimodal contrastive not yet implemented. specify data type for right")

    if ("validate" in args.mode or "evaluate" in args.mode) and args.target == "":
        raise ValueError("you must specify a target for validation/evaluation modes")

    dataset = datasets.TCGADataset(lr_data=[args.left, args.right], target=args.target) # TODO: splitting done here, not datasets

    if validate in args.mode:
        encoder = model.TCGAEncoders(data_types=[args.left], dataset=dataset, mode=args.mode)[0]
    else:
        encoders = model.TCGAEncoders(data_types=[args.left, args.right], dataset=dataset, mode=args.mode)

    if validate in args.mode:
        if split == "":
            split = [0.8, 0.2]
        else:
            split = string_to_list(args.mode)
            split = [float(spl) for spl in split]
            assert(len(split) == 2)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, split)
    else:
        if split == "":
            split = [0.6, 0.2, 0.2]
        else:
            split = string_to_list(args.mode)
            split = [float(spl) for spl in split]
            assert(len(split) == 3)
        pretrain_dataset, train_dataset, test_dataset = torch.utils.data.random_split(dataset, split)

    # TODO: save generated dataset & load old dataset if not pretraining mode

    if "validate" in args.mode:
        encoder = validate(args, encoder, train_datset, test_dataset)
    if "pretrain" in args.mode:
        encoders = pretrain(args, encoders, pretrain_dataset)
    if "finetune" in args.mode:
        encoders = finetune(args, encoders, train_dataset, test_dataset)
    if "inverse" in args.mode:
        encoders = inverse(args, encoders, train_dataset, test_dataset)

    if not args.silent:
        for i in range(0,15):
            subprocess.run("echo $\'\a\'", shell=True)
            time.sleep(3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--mode', default='(validate,pretraining,eval)', type=str)

    """
        GPU
        which gpu is used (on plasmon-ms, gpus 3 through 7 are free?)
    """
    parser.add_argument('--gpu', default=4, type=int)

    # Data generation options
    parser.add_argument('--left', default='', type=str) # tcga data type for left (primary) encoder
    parser.add_argument('--right', default='', type=str) # tcga data type for right (secondary) encoder
    parser.add_argument('--target', default='', type=str) # target for validation/evaluation task

    parser.add_argument('--split', default="", type=str) # pretrain/train/test split

    # File I/O
    """
        PATH_DIR
        directory in which to save the results from this experiment
    """
    parser.add_argument('--path_dir', default='', type=str)

    """
        EVIN_EPOCH
        which epoch to use for the eval/inverse loops

        -1 defaults to the last epoch; you can choose others or pick a range
        (the latter is not implemented yet)
    """
    parser.add_argument('--evin_epoch', default=-1, type=int)

    # Training reporting options
    parser.add_argument('--progress_every', default=5, type=int)
    parser.add_argument('--save_every', default=20, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--silent', default=False, action='store_true')
    parser.add_argument('--print_results', default=False, action='store_true')

    # Optimizer options
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--freeze_model', default=True, action='store_false')

    ## UNCOMMENT/IMPLEMENT THESE LATER FOR HYPERPARAMETER TUNING
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)

    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--pred_lr', default=0.02, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--sup_loss', default='mse', type=str)
    parser.add_argument('--cosine', default=True, action='store_false')

    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--clip', default=-1.0, type=float)

    # NN size options
    parser.add_argument('--repr_dim', default=32, type=int)

    args = parser.parse_args()
    main(args)

























dataset = datasets.GeneticClinicalDataset()
train_size = int(0.7 * len(dataset))
val_size = int(0.33 * (len(dataset) - train_size))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=256,
    **dataloader_kwargs
)

num_points = len(dataset[0][0].tolist())
assert(num_points == 2289)
num_samples = len(dataset)
assert(num_samples == 8839)

num_tests = len(dataset[0][2])
assert(num_tests == 2)
total_test_points = list(torch.cat(dataset[0][2]).size())[0]
assert(total_test_points == 26)
test_indices = [0] + [list(torch.cat(dataset[0][2][:x]).size())[0] for x in range(1, num_tests + 1)]
assert(test_indices == [0, 25, 26])

test_types = []
for i in range(num_tests):
    test_data = dataset[:][2][i].numpy()
    if np.array_equal(np.unique(test_data), np.array([0.])) or np.array_equal(np.unique(test_data), np.array([0, 1])):
        if np.array_equal(np.unique(np.sum(test_data, axis=1)), np.array([1])):
            test_types.append("categorical")
        elif np.size(test_data) / num_samples == 1:
            test_types.append("binary")
        else:
            test_types.append("vertical")
    else:
        test_types.append("numerical")
assert(test_types == ['categorical', 'binary'])

neural_model = nn.Sequential(
    nn.Linear(num_points, total_test_points)
)
optimizer = torch.optim.SGD(
    neural_model.parameters(),
    momentum=0.9,
    lr=0.005,
    weight_decay=0.0001
)
loss = torch.nn.MSELoss()

l2_model = sklearn.linear_model.Ridge()
l1_model = sklearn.linear_model.Lasso()
logistic_model = multioutput.MultiOutputRegressor(sklearn.linear_model.LogisticRegression(solver='liblinear'))
forest_model = ensemble.RandomForestRegressor()
sklearn_models = [l2_model, l1_model, logistic_model, forest_model]

# Neural train
for e in range(1, 31):
    neural_model.train()

    for it, batch in enumerate(train_loader):
        neural_model.zero_grad()

        out = neural_model(batch[:][0])
        target = torch.cat(batch[:][2], axis=1)

        l = loss(out, target)
        l.backward()
        optimizer.step()

# Simple models
for model in sklearn_models:
    model.fit(train_dataset[:][0], torch.cat(train_dataset[:][2], axis=1))

# Threshold setting with validation dataset
target = torch.cat(val_dataset[:][2], axis=1).cpu().numpy()

with torch.no_grad():
    neural_model.eval()

    neural_out = neural_model(val_dataset[:][0]).cpu().numpy()
    print(neural_out.shape)

val_numpy = val_dataset[:][0].cpu().numpy()
sklearn_outs = []
for model in sklearn_models:
    sklearn_outs.append(model.predict(val_numpy))

neural_thresh = []
sklearn_threshs = []
for model in sklearn_models:
    sklearn_threshs.append([])

for it, type in enumerate(test_types):
    if type == "binary":
        target_type = target[:, test_indices[it]:test_indices[it + 1]].flatten()
        for it2, out in enumerate([neural_out] + sklearn_outs):
            out_type = out[:, test_indices[it]:test_indices[it + 1]].flatten()


            if np.max(np.abs(out_type - np.round(out_type))) < 0.02:
                thresh = 0.5
            else:
                positive = np.sort(out_type[target_type == 1])
                negative = np.sort(np.concatenate((out_type[target_type == 0], np.full((1,), np.inf))))

                idx_negative = np.searchsorted(positive, negative)
                idx_negative = np.arange(len(idx_negative)) + np.full(np.shape(idx_negative), len(positive)) - idx_negative
                thresh = negative[np.argmax(idx_negative)]

            if it2 == 0:
                neural_thresh.append(thresh)
            else:
                sklearn_threshs[it2 - 1].append(thresh)
    elif type == "vertical":
        raise NotImplementedError("Thresholds for data type vertical not yet developed")
    else:
        neural_thresh.append(None)
        sklearn_threshs = [x + [None] for x in sklearn_threshs]

# Testing
target = torch.cat(test_dataset[:][2], axis=1).cpu().numpy()

with torch.no_grad():
    neural_model.eval()

    neural_out = neural_model(test_dataset[:][0]).cpu().numpy()

test_numpy = test_dataset[:][0].cpu().numpy()
sklearn_outs = []
for model in sklearn_models:
    sklearn_outs.append(model.predict(test_numpy))

for it, type in enumerate(test_types):
    for it2, (out, thresh) in enumerate(zip([neural_out] + sklearn_outs, [neural_thresh] + sklearn_threshs)):
        target_type = target[:, test_indices[it]:test_indices[it + 1]]
        out_type = out[:, test_indices[it]:test_indices[it + 1]]

        if type == "binary":
            acc = np.size(np.where((out_type > thresh[it]) == target_type)[0]) / np.size(out_type)
            bare_acc = max(np.mean(target_type), 1 - np.mean(target_type))
        elif type == "categorical":
            out_type = np.argmax(out_type, axis=1)
            target_type = np.argmax(target_type, axis=1)
            acc = np.size(np.where(out_type == target_type)[0]) / np.size(out_type)
            bare_acc = np.max(np.bincount(target_type)) / np.size(target_type)
        elif type == "numerical":
            out_type = np.mean(np.abs(out_type - target_type))
            acc = out_type
            bare_acc = np.mean(np.abs(target_type - np.mean(target_type)))
        elif type == "vertical":
            out_type = out_type[out_type > np.reshape(thresh[it], (1, -1))]
            acc = np.where(out_type == target_type)[0] / np.size(out_type)
            bare_acc = np.sum(np.maximum(np.mean(target_type, axis=0), 1 - np.mean(target_type, axis=0)))
        print(bare_acc)
        print(acc)



#for test_output in range(2, len(args.dataset[0].tolist())):
"""for test_output in range(2, num_tests + 2):
    num_points = len(dataset[0][0].tolist())
    print(num_points)

    model = nn.Sequential(
        nn.Linear(num_points, 1)
    )

    # optimization
    optimizer = torch.optim.SGD(
        model.parameters(),
        momentum=0.9,
        lr=0.005,
        weight_decay=0.0001
    )

    for e in range(1, 101):
        model.train()

        ls = []
        for it, elem in enumerate(train_loader):
            model.zero_grad()

            out = model(elem[0])
            target = elem[test_output]
            #print(torch.sum(target, axis=1))
            loss = torch.nn.MSELoss()
            l = loss(out, target)
            ls.append(l.detach().cpu().numpy())

            l.backward()
            optimizer.step()
        print(e)
        print(np.mean(np.array(ls)))

        with torch.no_grad():
            acc = 0
            total = 0
            #for it, elem in enumerate(test_loader):
            #    out = np.argmax(model(elem[0]).cpu().numpy())
            #    target = np.argmax(elem[2].cpu().numpy())
            #    if out == np.argmax(elem[2].cpu().numpy()):
            #        acc = acc + 1
            #    total = total + 1
            errors = []
            for it, elem in enumerate(test_loader):
                out = model(elem[0]).cpu().numpy()
                target = elem[3].cpu().numpy()
                errors.append(np.mean(np.square(out - target)))
            #print(acc / total)
            print(np.sqrt(np.mean(np.array(errors))))"""

"""def supervised_loop(args):
    train_size = int(0.8 * len(args.dataset))
    test_size = len(args.dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(args.dataset, [train_size, test_size])

    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )

    #for test_output in range(2, len(args.dataset[0].tolist())):
    for test_output in range(2, 3):
        num_points = len(args.dataset[0][0].tolist())
        print(num_points)

        model = nn.Sequential(
            nn.Linear(num_points, 1)
        )

        # optimization
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=0.9,
            lr=args.lr,
            weight_decay=args.wd
        )

        for e in range(1, args.epochs + 1):
            model.train()

            for it, elem in enumerate(train_loader):
                model.zero_grad()

                loss = torch.nn.MSELoss(model(elem[0]), elem[test_output])

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                acc = 0
                total = 0
                for it, elem in enumerate(test_loader):
                    out = np.argmax(model(elem[0]).cpu().numpy())
                    if out == np.argmax(elem[2].cpu().numpy()):
                        acc = acc + 1
                    total = total + 1
                print(acc / total)


def training_loop(args, encoder=None):
    # dataset
    dataloader_kwargs = dict(drop_last=True, pin_memory=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(
        dataset=PendulumImageDataset(size=args.data_size, trajectory_length=args.traj_len, nnoise=args.nnoise, gnoise=args.gnoise,
                                        img_size=args.img_size, diff_time=args.diff_time, gaps=args.gaps, crop=args.crop, crop_c = args.crop_c,
                                        t_window=args.t_window, t_range=args.t_range, mink=args.mink, maxk=args.maxk),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )
    if args.validation:
        test_loader = torch.utils.data.DataLoader(
            dataset=PendulumImageDataset(size=512, gaps=args.gaps),
            shuffle=False,
            batch_size=512,
            **dataloader_kwargs
        )
    if verbose:
        print("[Self-supervised] Completed data loading")

    # model
    main_branch = Branch(args.repr_dim, deeper=args.deeper, affine=args.affine, encoder=encoder)
    if torch.cuda.is_available():
        main_branch.cuda()

    if args.method == "simsiam":
        h = PredictionMLP(args.repr_dim, args.dim_pred, args.repr_dim)
        if torch.cuda.is_available():
            h.cuda()

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr,
        weight_decay=args.wd
    ) # extremely sensitive to learning rate
    lr_scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=0,
        num_epochs=args.epochs,
        base_lr=args.lr * args.bsz / 256,
        final_lr=0,
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True
    )
    if args.method == "simsiam":
        pred_optimizer = torch.optim.SGD(
            h.parameters(),
            momentum=0.9,
            lr=args.pred_lr,
            weight_decay=args.wd
        )

    # macros
    b = main_branch.encoder
    proj = main_branch.projector

    # helpers
    def get_z(x):
        return proj(b(x))

    def apply_loss(z1, z2, distance):
        #if args.loss == 'square':
        #    loss = (z1 - z2).pow(2).sum()
        if args.method == 'infonce':
            loss = 0.5 * info_nce(z1, z2, temperature=args.temp, distance=distance) + 0.5 * info_nce(z2, z1, temperature=args.temp, distance=distance)
        elif args.method == 'simsiam':
            p1 = h(z1)
            p2 = h(z2)
            loss = simsiam_loss(p1, z2, distance=distance) / 2 + simsiam_loss(p2, z1, distance=distance) / 2
        return loss

    if verbose:
        print("[Self-supervised] Model generation complete, training begins")

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'training_loss.log'), 'w')
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, '0.pth'))

    data_args = vars(args)

    with open(os.path.join(args.path_dir, '0.args'), 'w') as fp:
        json.dump(data_args, fp, indent=4)

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()
        if args.method == "simsiam":
            h.train()

        # epoch
        losses = []
        zmin = torch.tensor(10000)
        zmax = torch.tensor(-10000)
        for it, (x1, x2, energy, q1, q2) in enumerate(train_loader):
            # zero grad
            main_branch.zero_grad()
            if args.method == "simsiam":
                h.zero_grad()

            # forward pass
            z1 = get_z(x1)
            z2 = get_z(x2)

            zmin = torch.min(torch.min(z1, zmin))
            zmax = torch.max(torch.max(z1, zmax))
            zmin = torch.min(torch.min(z2, zmin))
            zmax = torch.max(torch.max(z2, zmax))
            if args.cosine:
                loss = apply_loss(z1, z2, distance="cosine")
            else:
                loss = apply_loss(z1, z2, distance="euclidean")
            losses.append(loss.item())

            # optimization step
            loss.backward()
            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(main_branch.parameters(), args.clip)
                if args.method == "simsiam":
                    torch.nn.utils.clip_grad_norm_(h.parameters(), args.clip * 2)
            optimizer.step()
            lr_scheduler.step()
            if args.method == "simsiam":
                pred_optimizer.step()

        if e % args.progress_every == 0 or e % args.save_every == 0:
            if args.validation:
                main_branch.eval()
                val_loss = -1
                for it, (x1, x2, energy, q1, q2) in enumerate(test_loader):
                    val_loss = loss(get_z(x1), get_z(x2)).item()
                    break
                line_to_print = f'epoch: {e} | loss: {loss.item()} | val loss: {val_loss.item()} | time_elapsed: {time.time() - start:.3f}'
            else:
                losses = torch.tensor(losses)
                losses = torch.std(losses)
                zrange = zmax - zmin
                line_to_print = f'epoch: {e} | loss: {loss.item()} | std: {losses.item()} | range: {zrange} | time_elapsed: {time.time() - start:.3f}'

            print(line_to_print)

            if e % args.save_every == 0:
                torch.save(dict(epoch=0, state_dict=main_branch.state_dict()), os.path.join(args.path_dir, f'{e}.pth'))
                file_to_update.write(line_to_print + '\n')
                file_to_update.flush()

                with open(os.path.join(args.path_dir, f'{e}.args'), 'w') as fp:
                    json.dump(data_args, fp, indent=4)
                print("[saved]")

    file_to_update.close()

    if verbose:
        print("[Self-supervised] Training complete")

    return main_branch.encoder


def testing_loop(args, encoder=None):
    global verbose

    load_files = args.load_file
    if args.load_every != -1:
        load_files = []
        idx = 0
        while 1 + 1 == 2 and idx * args.load_every <= args.load_max:
            file_to_add = os.path.join(args.path_dir, str(idx * args.load_every) + ".pth")
            if os.path.isfile(file_to_add):
                load_files.append(file_to_add)
                idx = idx + 1
            else:
                break
    elif load_files == "recent":
        load_files = [most_recent_file(args.path_dir, ext=".pth")]
    else:
        load_files = os.path.join(args.path_dir, load_files)
        load_files = [load_files]

    b = []
    data_args = {}

    for load_file in load_files:
        with open(load_file[:-4] + ".args", 'rb') as fp:
            new_data_args = json.load(fp)
            if data_args == {}:
                data_args = new_data_args
            else:
                assert(data_args == new_data_args)

        branch = Branch(data_args['repr_dim'], deeper=args.deeper, affine=args.affine, encoder=encoder)
        branch.load_state_dict(torch.load(load_file)["state_dict"])

        if torch.cuda.is_available():
            branch.cuda()

        branch.eval()
        b.append(branch.encoder)

    if verbose:
        print("[testing] Completed model loading")

    #test_k2, test_data, test_q = pendulum_train_gen(data_size=args.data_size,
        #traj_samples=(args.traj_len if args.sparse_testing else 1),
        #noise=args.noise, full_out=args.sparse_testing,
        #uniform=(not args.sparse_testing),img_size=args.img_size,
        #diff_time=args.diff_time, gaps=args.gaps,crop=args.crop,
        #crop_c = args.crop_c, t_window=args.t_window,t_range=args.t_range,
        #mink=args.mink, maxk=args.max)
    if not args.use_training_data:
        test_k2, test_data, test_q = pendulum_train_gen(data_size=args.data_size,
            traj_samples=args.traj_len,
            nnoise=args.nnoise, gnoise=args.gnoise, gaps=args.gaps, uniform=True, img_size=data_args['img_size'],
            diff_time=data_args['diff_time'], crop=data_args['crop'], crop_c=data_args['crop_c'])
    else:
        test_data = np.load(os.path.join(args.path_dir, "training_data.npy"))
        test_k2 = np.load(os.path.join(args.path_dir, "training_k2.npy"))
        test_q = np.load(os.path.join(args.path_dir, "training_q.npy"))

    if verbose:
        print("[testing] Completed data loading")

    coded = np.zeros((len(b), args.data_size, args.traj_len, data_args["repr_dim"]))

    if torch.cuda.is_available():
        for i in range(len(b)):
            if verbose:
                print("[testing] testing " + load_files[i])
            for j in range(0, args.data_size):
                coded[i, j, :, :] = b[i](torch.FloatTensor(test_data[j, :, :, :, :]).cuda()).cpu().detach().numpy()
    else:
        for i in range(len(b)):
            for j in range(0, args.data_size):
                coded[i, j, :, :] = b[i](torch.FloatTensor(test_data[j, :, :, :, :])).detach().numpy()
    energies = test_k2[:, 0]
    qs = test_q

    os.makedirs(os.path.join(args.path_dir, "testing"), exist_ok=True)
    for idx, load_file in enumerate(load_files):
        save_file = "-" + load_file.rpartition("/")[2].rpartition(".")[0]
        np.save(os.path.join(args.path_dir, "testing/coded" + save_file + ".npy"), coded[idx])
    np.save(os.path.join(args.path_dir, "testing/energies.npy"), energies)
    np.save(os.path.join(args.path_dir, "testing/qs.npy"), qs)

    if verbose:
        print("[testing] Testing data saved.")


def analysis_loop(args, encoder=None):
    global verbose

    load_files = args.load_file
    if args.load_every != -1:
        load_files = []
        idx = 0
        while 1 + 1 == 2 and idx * args.load_every <= args.load_max:
            file_to_add = os.path.join(args.path_dir, str(idx * args.load_every) + ".pth")
            if os.path.isfile(file_to_add):
                load_files.append(file_to_add)
                idx = idx + 1
            else:
                break
    elif load_files == "recent":
        load_files = [most_recent_file(args.path_dir, ext=".pth")]
    else:
        load_files = os.path.join(args.path_dir, load_files)
        load_files = [load_files]

    b = []
    data_args = {}

    for load_file in load_files:
        with open(load_file[:-4] + ".args", 'rb') as fp:
            new_data_args = json.load(fp)
            if data_args == {}:
                data_args = new_data_args
            else:
                assert(data_args == new_data_args)

        branch = Branch(data_args['repr_dim'], deeper=args.deeper, affine=args.affine, encoder=encoder)
        branch.load_state_dict(torch.load(load_file)["state_dict"])

        if torch.cuda.is_available():
            branch.cuda()

        branch.eval()
        b.append(branch.encoder)

    if verbose:
        print("[analysis] Completed model loading")

    # Analysis: NN modeling, density, spearman, line optimality

    prt_image = data_args['crop'] != 1.0
    prt_time = data_args['t_window'] != [-1, -1] or data_args['t_range'] != -1
    prt_energy = data_args['gaps'] != [-1, -1] or data_args['mink'] != 0 or data_args['maxk'] != 1

    if "both_noise" in data_args.keys():
        data_args["nnoise"] = data_args["noise"]
        data_args["gnoise"] = data_args["noise"]

    def slim(arr):
        shape = list(arr.shape)
        shape[0] = shape[0] * shape[1]
        shape.pop(1)
        return np.reshape(arr, tuple(shape))

    def segment_analysis(borders, q_borders, data, k2, q, print_results=args.print_results):
        k2 = k2.ravel()
        idx = np.searchsorted(borders, k2).ravel()
        q_idx = np.searchsorted(q_borders, q).ravel()

        fine_density = np.zeros((len(b), len(borders) + 1, len(q_borders) + 1))
        fine_spearman = np.zeros((len(b), len(borders) + 1, len(q_borders) + 1))
        q_density = np.zeros((len(b), len(q_borders) + 1))
        q_spearman = np.zeros((len(b), len(q_borders) + 1))
        k_density = np.zeros((len(b), len(borders) + 1))
        k_spearman = np.zeros((len(b), len(borders) + 1))
        full_dn = np.zeros((len(b)))
        full_sm = np.zeros((len(b)))

        data = torch.FloatTensor(data)
        if torch.cuda.is_available():
            data = data.cuda()

        lens = [borders[0]] + list(np.array(borders[1:]) - np.array(borders[:-1])) + list(1 - np.array([borders[-1]]))
        lens = np.array(lens)
        q_lens = [q_borders[0] + 3.1415] + list(np.array(q_borders[1:]) - np.array(q_borders[:-1])) + list(3.1415 - np.array([q_borders[-1]]))
        q_lens = np.array(q_lens)

        for i in range(0, len(b)):
            b[i].eval()
            with torch.no_grad():
                if args.use_training_data:
                    out = torch.zeros((data.shape[0], data_args["repr_dim"]))
                    shp = list(data.shape)
                    shp.insert(0, shp[0] // 16)
                    shp[1] = 16
                    data = torch.reshape(data, shp)
                    for j in range(shp[0]):
                        out[16 * j:16 * (j + 1)] = b[i](data[j])
                else:
                    out = b[i](data)
                out = out.cpu().detach().numpy()

                if np.size(out, axis=1) > 1:
                    spectral = SpectralEmbedding(n_components=1)
                    out = spectral.fit_transform(out)

                full_sm[i] = stats.spearmanr(out.ravel(), k2).correlation
                full_dn[i] = (np.quantile(out, 0.75) - np.quantile(out, 0.25)) / (0.5 * np.sum(lens))

                for j in range(0, len(q_borders) + 1):
                    seg_out = out[q_idx == j]
                    seg_k2 = k2[q_idx == j]
                    if np.size(seg_out, axis=0) <= 1:
                        continue

                    q_spearman[i, j] = stats.spearmanr(seg_out.ravel(), seg_k2).correlation
                    q_density[i, j] = np.reciprocal((np.quantile(seg_out, 0.75) - np.quantile(seg_out, 0.25)) / (0.5 * q_lens[j]))

                for j in range(0, len(borders) + 1):
                    seg_out = out[idx == j]
                    seg_k2 = k2[idx == j]
                    if np.size(seg_out, axis=0) <= 1:
                        continue

                    k_spearman[i, j] = stats.spearmanr(seg_out.ravel(), seg_k2).correlation
                    k_density[i, j] = np.reciprocal((np.quantile(seg_out, 0.75) - np.quantile(seg_out, 0.25)) / (0.5 * lens[j]))

                    for k in range(0, len(q_borders) + 1):
                        good_idx = np.ones(idx.shape)
                        good_idx[idx != j] = 0
                        good_idx[q_idx != k] = 0
                        seg_out = out[good_idx == 1]
                        seg_k2 = k2[good_idx == 1]
                        if np.size(seg_out, axis=0) <= 5:
                            continue

                        fine_spearman[i, j, k] = stats.spearmanr(seg_out.ravel(), seg_k2).correlation
                        fine_density[i, j, k] = np.reciprocal((np.quantile(seg_out, 0.75) - np.quantile(seg_out, 0.25)) / (0.5 * lens[j] * q_lens[k]))

        if print_results:
            file_names = []
            for i in range(0, len(b)):
                file_name = load_files[i][:-4]
                file_name = file_name[file_name.rfind('/') + 1:]
                file_names.append(file_name)

            full_dn = full_dn.round(decimals=5)
            full_sm = full_sm.round(decimals=5)
            k_density = k_density.round(decimals=5)
            k_spearman = k_spearman.round(decimals=5)
            q_density = q_density.round(decimals=5)
            q_spearman = q_spearman.round(decimals=5)
            fine_density = fine_density.round(decimals=5)
            fine_spearman = fine_spearman.round(decimals=5)

            print("density")
            for i in range(0, len(b)):
                print(file_names[i] + " " + str(full_dn[i]) + " " + np.array2string(k_density[i]) + " " + np.array2string(q_density[i]))
                print(np.array2string(fine_density[i]))
            print("")
            print("spearman")
            for i in range(0, len(b)):
                print(file_names[i] + " " + str(full_sm[i])+ " " + np.array2string(k_spearman[i]) + " " + np.array2string(q_spearman[i]))
                print(np.array2string(fine_spearman[i]))
            print("")
            return
        else:
            return [full_dn.tolist(), full_sm.tolist(), k_density.tolist(), k_spearman.tolist(), q_density.tolist(), q_spearman.tolist(), fine_density.tolist(), fine_spearman.tolist()]

    if not args.use_training_data:
        same_k2, same_data, same_q = pendulum_train_gen(data_size=args.data_size,traj_samples=1,
            nnoise=data_args['nnoise'],uniform=True,img_size=data_args['img_size'],gnoise=data_args['gnoise'],
            diff_time=data_args['diff_time'], gaps=args.gaps,
            crop=data_args['crop'],crop_c = data_args['crop_c'],
            t_window=data_args['t_window'],t_range=data_args['t_range'],
            mink=args.mink, maxk=args.maxk)
    else:
        same_data = np.load(os.path.join(args.path_dir, "training_data.npy"))
        same_k2 = np.load(os.path.join(args.path_dir, "training_k2.npy"))
        same_q = np.load(os.path.join(args.path_dir, "training_q.npy"))

    same_k2 = slim(same_k2[:,:,0])
    same_q = slim(same_q[:,:,0])
    print(np.sum(same_data[:,:,0,15:,15:]))
    same_data = slim(same_data)

    to_save = []

    num_gaps = data_args['gaps'][0]
    if num_gaps == -1:
        num_gaps = 0

    num_gaps = int(num_gaps)

    if num_gaps > 0:
        gap_width = data_args['gaps'][1] / num_gaps
        print(gap_width)
        btwn_width = (1 - data_args['gaps'][1]) / (num_gaps + 1)
    else:
        btwn_width = 1/7
        gap_width = 1/7
        num_gaps = 3
    btwn_width = btwn_width * (data_args['maxk'] - data_args['mink'])
    gap_width = gap_width * (data_args['maxk'] - data_args['mink'])

    borders = [data_args['mink']]

    for i in range(0, num_gaps):
        borders.append(borders[-1] + btwn_width)
        borders.append(borders[-1] + gap_width)
    borders.append(borders[-1] + btwn_width)

    q_borders = np.array([0,1/7,2/7,3/7,4/7,5/7,6/7,1])
    q_borders = np.min(same_q) + q_borders * (np.max(same_q) - np.min(same_q))

    print("same")
    same_args = deepcopy(data_args)
    same_args["test_mode"] = "same"
    if not args.print_results:
        to_save.append([same_args] + segment_analysis(borders, q_borders, same_data, same_k2, same_q))
    else:
        segment_analysis(borders, q_borders, same_data, same_k2, same_q)

    if prt_image:
        red_visible = same_data[:, 2, :, :]
        red_visible = np.reshape(red_visible, (np.size(red_visible, axis=0), -1))
        red_visible = np.where(red_visible == 0, red_visible + 0.0001, red_visible)
        red_visible = np.sum(2 - np.ceil(red_visible * 2), axis=1)
        red_visible = np.where(red_visible < 9, 0, 1)

        blue_visible = same_data[:, 0, :, :]
        blue_visible = np.reshape(blue_visible, (np.size(blue_visible, axis=0), -1))
        blue_visible = np.where(blue_visible == 0, blue_visible + 0.0001, blue_visible)
        blue_visible = np.sum(2 - np.ceil(blue_visible * 2), axis=1)
        blue_visible = np.where(blue_visible < 9, 0, 1)

        visible = red_visible * blue_visible

        vis_data = same_data[visible == 1, ...]
        vis_k2 = same_k2[visible == 1, ...]
        vis_q = same_q[visible == 1, ...]

        visq_borders = np.array([0,1/7,2/7,3/7,4/7,5/7,6/7,1])
        visq_borders = np.min(vis_q) + visq_borders * (np.max(vis_q) - np.min(vis_q))

        vis_args = deepcopy(data_args)
        vis_args["test_mode"] = "visible"
        if not args.print_results:
            to_save.append([vis_args] + segment_analysis(borders, visq_borders, vis_data, vis_k2, vis_q))
        else:
            segment_analysis(borders, visq_borders, vis_data, vis_k2, vis_q)

    if prt_time:
        print("all times")
        all_k2, all_data, all_q = pendulum_train_gen(data_size=args.data_size,traj_samples=1,
            nnoise=data_args['nnoise'],uniform=True,img_size=data_args['img_size'],
            diff_time=data_args['diff_time'], gaps=data_args['gaps'],
            crop=data_args['crop'],crop_c = data_args['crop_c'],
            t_window=[-1,-1],t_range=-1,gnoise=data_args['gnoise'],
            mink=data_args['mink'], maxk=data_args['maxk'])

        all_k2 = slim(all_k2[:,:,0])
        all_data = slim(all_data)
        all_q = slim(all_q[:,:,0])

        allq_borders = np.array([0,1/7,2/7,3/7,4/7,5/7,6/7,1])
        allq_borders = np.min(all_q) + allq_borders * (np.max(all_q) - np.min(all_q))

        time_args = deepcopy(data_args)
        time_args["t_window"] = [-1,-1]
        time_args["t_range"] = -1
        time_args["test_mode"] = "time"
        if not args.print_results:
            to_save.append([time_args] + segment_analysis(borders, allq_borders, all_data, all_k2, all_q))
        else:
            segment_analysis(borders, allq_borders, all_data, all_k2, all_q)

    if prt_energy:
        print("all energies")
        all_k2, all_data, all_q = pendulum_train_gen(data_size=args.data_size,traj_samples=1,
            nnoise=data_args['nnoise'],uniform=True,img_size=data_args['img_size'],
            diff_time=data_args['diff_time'], gaps=[-1,-1],gnoise=data_args['gnoise'],
            crop=data_args['crop'],crop_c = data_args['crop_c'],
            t_window=data_args['t_window'],t_range=data_args['t_range'],
            mink=0, maxk=1)

        all_k2 = slim(all_k2[:,:,0])
        all_data = slim(all_data)
        all_q = slim(all_q[:,:,0])

        new_borders = np.array([0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1])

        allq_borders = np.array([0,1/7,2/7,3/7,4/7,5/7,6/7,1])
        allq_borders = np.min(all_q) + allq_borders * (np.max(all_q) - np.min(all_q))

        energy_args = deepcopy(data_args)
        energy_args["gaps"] = [-1,-1]
        energy_args["mink"] = 0
        energy_args["maxk"] = 1
        energy_args["test_mode"] = "energy"
        if not args.print_results:
            to_save.append([energy_args] + segment_analysis(new_borders, allq_borders, all_data, all_k2, all_q))
        else:
            segment_analysis(new_borders, allq_borders, all_data, all_k2, all_q)

    if args.nnoise != data_args["nnoise"] or args.gnoise != data_args["gnoise"]:
        print("args-specified noise")
        noise_k2, noise_data, noise_q = pendulum_train_gen(data_size=args.data_size,traj_samples=1,
            nnoise=args.nnoise,uniform=True,img_size=data_args['img_size'],
            diff_time=data_args['diff_time'], gaps=data_args['gaps'],
            crop=data_args['crop'],crop_c = data_args['crop_c'],gnoise=args.gnoise,
            t_window=data_args['t_window'],t_range=data_args['t_range'],
            mink=data_args['mink'], maxk=data_args['maxk'])

        noise_k2 = slim(noise_k2[:,:,0])
        noise_data = slim(noise_data)
        noise_q = slim(noise_q[:,:,0])

        noise_args = deepcopy(data_args)
        noise_args["nnoise"] = args.nnoise
        noise_args["test_mode"] = "noise"
        if not args.print_results:
            to_save.append([noise_args] + segment_analysis(borders, q_borders, noise_data, noise_k2, noise_q))
        else:
            segment_analysis(borders, noise_data, noise_k2)

    if not os.path.isfile("data/master_experiments.json"):
        open("data/master_experiments.json", "w")

    if not args.print_results:
        with open("data/master_experiments.json", "r+") as fp:
            if os.stat("data/master_experiments.json").st_size == 0:
                all_experiments = {}
            else:
                all_experiments = json.load(fp)

            for i in range(0, len(b)):
                file_name = load_files[i][:-4]
                file_name = file_name[file_name.rfind('/') + 1:]

                experiment_name = load_files[i]
                experiment_name = experiment_name[:experiment_name.rfind('/')]
                experiment_name = experiment_name[experiment_name.rfind('/') + 1:]

                for instance in to_save:
                    dict = {}
                    dict["experiment_name"] = experiment_name
                    dict["old_params"] = data_args
                    dict["new_params"] = instance[0]
                    dict["epoch"] = file_name
                    dict["full_density"] = instance[1][i]
                    dict["full_spearman"] = instance[2][i]
                    dict["k_density"] = instance[3][i]
                    dict["k_spearman"] = instance[4][i]
                    dict["q_density"] = instance[5][i]
                    dict["q_spearman"] = instance[6][i]
                    dict["fine_density"] = instance[7][i]
                    dict["fine_spearman"] = instance[8][i]

                    is_duplicate = False
                    for key, value in all_experiments.items():
                        this_duplicate = True
                        for kk in ["experiment_name", "old_params", "new_params", "epoch"]:
                            if dict[kk] != value[kk]:
                                this_duplicate = False

                        if this_duplicate:
                            is_duplicate = True

                    if not is_duplicate:
                        all_experiments[str(datetime.datetime.now())] = dict
            fp.seek(0)
            json.dump(all_experiments, fp, indent=4)
            fp.truncate()

def main(args):
    global verbose

    if args.verbose:
        verbose = True

    args.gaps = args.gaps.split(",")
    assert(len(args.gaps) == 2)
    args.gaps[0] = float(args.gaps[0])
    args.gaps[1] = float(args.gaps[1])

    args.crop_c = args.crop_c.split(",")
    assert(len(args.crop_c) == 2)
    args.crop_c[0] = float(args.crop_c[0])
    args.crop_c[1] = float(args.crop_c[1])

    args.t_window = args.t_window.split(",")
    assert(len(args.t_window) == 2)
    args.t_window[0] = float(args.t_window[0])
    args.t_window[1] = float(args.t_window[1])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.path_dir == "":
        raise UserWarning("please do not pass empty experiment names")
    args.path_dir = '../output/pendulum/' + args.path_dir

    # TODO: add dataset arg & load here

    set_deterministic(42)
    if args.mode == 'training':
        training_loop(args)
    elif args.mode == 'testing':
        testing_loop(args)
    elif args.mode == 'plotting':
        plotting_loop(args)
    elif args.mode == 'analysis':
        analysis_loop(args)
    elif args.mode == 'supervised':
        supervised_loop(args)
    else:
        raise

    if not args.silent:
        for i in range(0,15):
            subprocess.run("echo $\'\a\'", shell=True)
            time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='training', type=str,
                        choices=['plotting', 'training', 'testing', 'analysis', 'supervised'])
    parser.add_argument('--method', default='infonce', type=str,
                        choices=['infonce', 'simsiam'])
    parser.add_argument('--gpu', default=4, type=int)

    # Data generation options
    parser.add_argument('--data_size', default=5120, type=int)
    parser.add_argument('--density', default=1000, type=int)
    parser.add_argument('--traj_len', default=20, type=int)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--diff_time', default=0.5, type=float)

    parser.add_argument('--save_training_data', default=False, action='store_true')
    parser.add_argument('--use_training_data', default=False, action='store_true')

    parser.add_argument('--gaps', default="-1,-1", type=str)
    parser.add_argument('--crop', default=1.0, type=float)
    parser.add_argument('--crop_c', default="-1,-1", type=str)
    parser.add_argument('--t_window', default="-1,-1", type=str)
    parser.add_argument('--t_range', default=-1,type=float)
    parser.add_argument('--mink', default=0.0, type=float)
    parser.add_argument('--maxk', default=1.0, type=float)

    parser.add_argument('--gnoise', default=0., type=float)
    parser.add_argument('--nnoise', default=0., type=float)

    # File I/O
    parser.add_argument('--path_dir', default='', type=str)

    parser.add_argument('--load_file', default='recent', type=str)
    parser.add_argument('--load_every', default='-1', type=int)
    parser.add_argument('--load_max', default=1000000, type=int)

    # Training reporting options
    parser.add_argument('--progress_every', default=5, type=int)
    parser.add_argument('--save_every', default=20, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--validation', default=False, action='store_true')
    parser.add_argument('--silent', default=False, action='store_true')
    parser.add_argument('--print_results', default=False, action='store_true')

    # Optimizer options
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bsz', default=512, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)

    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--pred_lr', default=0.02, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--sup_loss', default='mse', type=str)
    parser.add_argument('--cosine', default=False, action='store_true')

    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--clip', default=3.0, type=float)

    # NN size options
    parser.add_argument('--dim_pred', default=1, type=int)
    parser.add_argument('--repr_dim', default=1, type=int)
    parser.add_argument('--affine', action='store_false')
    parser.add_argument('--deeper', action='store_false')

    args = parser.parse_args()
    main(args)"""
