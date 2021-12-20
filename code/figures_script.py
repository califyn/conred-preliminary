import glob
import random
from lifelines import CoxPHFitter
import transformers
import model as fancy_model
import torch
import math
import traceback
from scipy.stats import gmean, linregress
import time
import subprocess
import argparse
try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    import matplotlib
    matplotlib.use('ps')
    pass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import json 
import scipy.stats
import os
from sklearn.metrics import RocCurveDisplay

os.environ["CUDA_VISIBLE_DEVICES"] = ''
print(torch.cuda.device_count())

def cinterval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def logFit(x,y):
    # cache some frequently reused terms
    sumy = np.sum(y)
    sumlogx = np.sum(np.log(x))

    b = (x.size*np.sum(y*np.log(x)) - sumy*sumlogx)/(x.size*np.sum(np.log(x)**2) - sumlogx**2)
    a = (sumy - b*sumlogx)/x.size

    return a,b

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_context("paper")

flare = sns.color_palette("flare", as_cmap=True)
cats = sns.color_palette("Set2")
gold = sns.color_palette("YlOrBr", as_cmap=True)
twoway = sns.diverging_palette(220, 20, as_cmap=True)

epoch = 1
my_prompts = False

base_exp_name = 'final'
base_models = []

twod_exp_name = '2d_e'
twod_models = []

small_exp_name = 'dimshrink'
small_models = []

site_exp_name = 'nosite'
site_models = []

data_exp_name = 'datashrink'
data_models = []

try_exp_name = "first_try"
try_models = []

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

def nandiv(a, b):
    if b != 0:
        return a / b
    else:
        return np.nan

def tokenize(s, lm_arch):
    if lm_arch == "distilbert":
        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    elif lm_arch == "biobert":
        tokenizer = transformers.AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    elif lm_arch == "bert":
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    if isinstance(s, str):
        s = [s]

    batch = tokenizer(s, padding='max_length', max_length=150, truncation=True, return_tensors="pt")

    return batch

def spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew[:, 3:]

def get_type(ids):
    df = pd.read_csv("../../tcga/new/huge_clinical_data.tsv", sep="\t")
    idx = [df["bcr_patient_barcode"].tolist().index(id) for id in ids]

    types = df["acronym"].tolist()
    set_type = sorted(list(set(types)))
    types = [set_type.index(x) for x in types]

    return np.array(types)[idx]

def get_surv(ids):
    df = pd.read_csv("../../tcga/new/survival_only.csv")
    idx = [df["bcr_patient_barcode"].tolist().index(id) for id in ids]

    ind = df["OS"].tolist()
    fine = df["OS.time"].tolist()

    return np.array(ind)[idx], np.array(fine)[idx]

def raw_clip_acc():
    # Figure 2
    # Reports raw accuracy and gain on the overall CLIP objective
    x = ["ConRED", "Random Type-Based", "Random"]
    y = [
            np.array([model["final_acc"] for model in base_models]),
            np.array([np.nansum(np.diagonal(model["type_confusion_matrix"])) / model["args"]["bsz"] for model in base_models]),
            np.full((len(base_models), ), 1 / base_models[0]["args"]["bsz"])
        ]

    print(y, "base accuracies: figure 2")
    plt.bar([0, 1, 2], [np.mean(k) for k in y], tick_label=x, color=[gold(0.5), gold(0.25), gold(0.25)])
    plt.errorbar(x, [np.mean(k) for k in y], yerr=np.array([cinterval(k) for k in y]), fmt='o', color='k')
    plt.ylabel('Top-1 retrieval accuracy')
    plt.vlines(x=1.8, ymin=np.mean(y[1]), ymax=np.mean(y[0]), colors=['k'])
    plt.hlines(y=np.mean(y[1]), xmin=1.75, xmax=1.85, colors=['k'])
    plt.hlines(y=np.mean(y[1]), xmin=1.4, xmax=1.75, colors=['k'], linestyles='dashed')
    plt.hlines(y=np.mean(y[0]), xmin=1.75, xmax=1.85, colors=['k'])
    plt.hlines(y=np.mean(y[0]), xmin=0.4, xmax=1.75, colors=['k'], linestyles='dashed')
    plt.text(1.9, 0.5 * (np.mean(y[1]) + np.mean(y[0])), str(round(float(np.mean(y[0]) / np.mean(y[1])), 1)) + "x Gain", verticalalignment='center')

    plt.show()

def type_match():
    # Figure 3
    # reports type identification accuracy
    x = base_models[0]["types"] 
    y = [np.array([model["type_confusion_matrix"][it, it] for model in base_models]) for it in range(len(x))]

    avg = [np.mean([model["real_class_sizes"][it] / np.sum(np.array(model["real_class_sizes"])) for model in base_models]) for it in range(len(x))]
    print(np.sum(np.array(avg)), "class sum = 1?")
    avg = [np.mean(y[it]) * avg[it] for it in range(len(x)) if not np.isnan(y[it]).any()]
    avg = np.sum(np.array(avg))

    x = [z for it, z in enumerate(x) if not np.isnan(y[it]).any()]
    y = [z for z in y if not np.isnan(z).any()]
    print('remember some types are nans and write in caption')

    x = [b for a, b in sorted(zip([-float(np.mean(k)) for k in y], x))]
    y = list(sorted(y, key=lambda k: -float(np.mean(k))))

    print(x, "types")
    print(y, "intertype accs: figure 3")

    plt.bar(list(range(len(x) + 1)), [np.mean(k) for k in y] + [avg], tick_label=x + ["Avg."], color=[cats[0]] * len(x) + [cats[1]])
    print(avg, "average acc")
    plt.errorbar(x, [np.mean(k) for k in y], yerr=np.array([cinterval(k) for k in y]), fmt='o', color='k')
    plt.ylabel("Type accuracy")
    plt.xticks(rotation=70)
    plt.ylim(0, 1.2)
    
    plt.show()

def inner_match():
    # Figure 4
    # Matching inside
    x = base_models[0]["types"]

    y = [np.array([nandiv(max(model["real_class_sizes"][it], 1), model["proj_class_sizes"][it]) for model in base_models]) for it in range(len(x))]

    x = [z for it, z in enumerate(x) if not np.isnan(y[it]).any()]
    y = [z for z in y if not np.isnan(z).any()]
    print('remember some types are nans and write in caption')

    x = [b for a, b in sorted(zip([-float(gmean(k)) for k in y], x))]
    y = list(sorted(y, key=lambda k: -float(gmean(k))))

    print(x, "types")
    print(y, "innertype accs: figure 4")

    print('mention that u did gmean not mean')
    num_bad = len([k for k in y if gmean(k) < 1])
    plt.bar(list(range(len(x))), [gmean(k) for k in y], tick_label=x, color=[cats[4]] * (len(x) - num_bad) + [cats[7]] * num_bad)
    plt.errorbar(x, [gmean(k) for k in y], yerr=np.array([cinterval(k) for k in y]), fmt='o', color='k')
    plt.axhline(y=1.0, linestyle='--', color='k', lw=1.0)
    plt.ylabel("Gain (vs. Type-Based)")
    ax = plt.gca()
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=70)
    ax.tick_params(axis=u'x', which=u'both',length=0)

    plt.show()

def acc_corr():
    # Figure 5
    # matching stuff
    i = base_models[0]["types"]
    x = [np.array([nandiv(max(model["real_class_sizes"][it], 1), model["proj_class_sizes"][it]) for model in base_models]) for it in range(len(i))]
    y = [np.array([model["type_confusion_matrix"][it, it] for model in base_models]) for it in range(len(i))]
    s = [np.array([nandiv(model["real_class_sizes"][it], np.sum(np.array(model["real_class_sizes"]))) for model in base_models]) for it in range(len(i))]

    good_idx = [it for it in range(len(i)) if (not np.isnan(x[it]).any()) and (not np.isnan(y[it]).any()) and (not np.isnan(s[it]).any())]
    i = [tt for it, tt in enumerate(i) if it in good_idx]
    x = np.array(x)[good_idx, :]
    y = np.array(y)[good_idx, :]
    s = np.array(s)[good_idx, :]

    plt.scatter([gmean(k) for k in x], [np.mean(k) for k in y], s=[np.mean(k) * 12000 for k in s], c=(cats * math.ceil(len(i) / 8))[:len(i)], alpha=0.7) 
    for it, t in enumerate(i):
        plt.text(gmean(x[it]), np.mean(y[it]), t, verticalalignment='center', horizontalalignment='center', fontsize=8)
    plt.xlabel("Gain (vs. Type-Based)")
    plt.ylabel("Type accuracy")
    plt.gca().set_ylim(top=1.1, bottom=0.65)

    plt.show()

def predictions(model):
    tasks = []
    for t in model["args"]["finetune"]:
        if t[-3:] == "_nl":
            tasks.append(t[:-3])

    tasks = sorted(list(set(tasks)))

    def normalize(arr):
        sizes = np.sum(np.square(arr), axis=1, keepdims=True)
        return arr / np.sqrt(sizes)


    if my_prompts:
        embed = normalize(model['rna-seq'])
    else:
        embed = np.expand_dims(normalize(model['rna-seq']), 1)
    for task in tasks:
        good_idx = np.where(model[task] != -1)[0]
        if not my_prompts:
            pembed = np.linalg.norm(model[task + "_pembed"], axis=1)
            nembed = np.linalg.norm(model[task + "_nembed"], axis=1)
            pembed = normalize(model[task + "_pembed"][good_idx, :])
            nembed = normalize(model[task + "_nembed"][good_idx, :])

        if my_prompts:
            task_embed = normalize(model["nl"](tokenize(prompts[task], model["args"]["lm_arch"])).detach().numpy())

        if not my_prompts:
            dist = np.matmul(embed[good_idx, :], np.stack((nembed, pembed), axis=2))
            dist = np.squeeze(dist, axis=1)
        else:
            dist = embed[good_idx, :] @ task_embed.T
        model[task + "_pred"] = np.full(model[task].shape, -1.0)
        model[task + "_pred"][good_idx] = 1/(1 + np.exp((dist[:, 1] - dist[:, 0]) / model["args"]["temp"]))
        print(task, model[task + "_pred"][:1000])

    return model

def ensemble(models):
    for model in models:
        for t in model["args"]["finetune"]:
            if t[-3:] == "_nl":
                tasks.append(t[:-3])

    tasks = sorted(list(set(tasks)))
    print(tasks, "Tasks")

    for task in tasks:
        test_targets = models[0][task][model["test_indicator"] == 1]

        test_preds = []
        for model in models:
            test_preds.append(model[task + "_pred"][model["test_indicator"] == 1])
        
        test_preds = np.array(test_preds)
        print(test_targets[:500].tolist())
        print(test_preds.shape, "preds 1")
        print(test_targets.shape, "targets 1")

        test_good_idx = np.where(test_targets != -1)[0]

        test_targets = test_targets[test_good_idx]
        test_preds = test_preds[:, test_good_idx]

        print(test_preds.shape, "preds 2")
        print(test_targets.shape, "targets 2")

        test_good_idx = ~np.any(np.isnan(test_preds), axis=0)

        test_targets = test_targets[test_good_idx]
        test_preds = test_preds[:, test_good_idx]

        print(test_preds.shape, "preds 3")
        print(test_targets.shape, "targets 3")

        # test plot
        test_cs = []
        test_cs.append(RocCurveDisplay.from_predictions(test_targets, np.quantile(test_preds, 0.5, axis=0)))
        for i in range(len(models)):
            test_cs.append(RocCurveDisplay.from_predictions(test_targets, test_preds[i, :]))
        for it, c in enumerate(test_cs[1:]):
            c.plot(name=str(it), ax=test_cs[0].ax_, color=cats[7])

        plt.title(f"test {task}")
        plt.show() 

def twod_surv_plot():
    model = twod_models[-1]

    pts = model["rna-seq"][model["test_indicator"] == 1]
    qts = model["reports"][model["test_indicator"] == 1]
    types = get_type([id for it, id in enumerate(model["ids"]) if model["test_indicator"][it] == 1])
    mask, leng = get_surv([id for it, id in enumerate(model["ids"]) if model["test_indicator"][it] == 1])
    print(mask.tolist().count(0), 'num 0')
    print(mask.tolist().count(1), 'num 1')
    idx = np.logical_and(mask == 0, ~np.isnan(leng))
    pts = pts[idx, :]
    qts = qts[idx, :]
    leng = leng[idx]
    types = [t for it, t in enumerate(types) if idx[it]]
    leng = leng / np.quantile(leng, 0.85)
    leng = np.minimum(np.full(leng.shape, 1.0), leng)
    print(leng[:50], "leng")
    
    print(max(types))
    plt.scatter(qts[:, 0], qts[:, 1], c=leng, marker="^", cmap=flare)
    plt.scatter(pts[:, 0], pts[:, 1], c=leng, cmap=flare)
    for i in range(max(types) + 1):
        choices = [it for it, t in enumerate(types) if t == i]
        plt.text(np.mean(pts[choices, 0]), np.mean(pts[choices, 1]), base_models[0]["types"][i])
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.show()

def site():
    sites = np.array([model["final_acc"] for model in base_models])
    nosites = np.array([model["final_acc"] for model in site_models])

    plt.bar([0, 1], [np.mean(sites), np.mean(nosites)], tick_label=["Normal", "With Site Effects"], color=[cats[0], cats[2]])  
    print([np.mean(sites), np.mean(nosites)], "mean acc w/o sites")
    plt.ylabel("Top-1 retrieval accuracy")
    plt.show()



def auc():
    arr = [[.683, .461, .705, .548, .579],
           [.671, .437, .682, .522, .563],
           [.7, .438, .712, .537, .543],
           [.756, .467, .653, .567, .29],
           [.699, .545, .722, .545, .609]]

    arr = np.array(arr).T
    plt.bar([0, 1, 2, 3, 4], [np.mean(arr[it]) for it in range(5)], tick_label=["Papillary subtype", "Lymphovascular invasion", "Perineural invasion", "Metastasis", "Ulceration"])
    plt.xticks(rotation=70)

    plt.show()


def data_size():
    models = data_models + base_models
    vals = []
    for it, model in enumerate(models):
        print(it, "model it")
        if model in base_models: 
            vals.append(1)
        elif model["var"] == "01":
            vals.append(0.125)
        elif model["var"] == "02":
            vals.append(0.25)
        elif model["var"] == "04":
            vals.append(0.5)
        else:
            raise ValueError('var doesnt work')
    vals = [x * np.size(base_models[0]["rna-seq"], axis=0) for x in vals]

    uvals = sorted(list(set(vals)))
    uaccs = []
    for u in uvals:
        umodels = [model for it, model in enumerate(models) if vals[it] == u]
        uaccs.append(np.array([umodel["final_acc"] for umodel in umodels]))

    plt.gca().set_xscale('log')
    lr = linregress(uvals, [np.mean(k) for k in uaccs])
    x = np.linspace(min(uvals) * 0.5, max(uvals) * 6) 
    y = x * lr.slope + lr.intercept
    a, b = logFit(np.array(uvals), np.array([np.mean(k) for k in uaccs]))
    yy = a + b * np.log(x) 
    plt.plot(uvals, [np.mean(k) for k in uaccs], c=cats[1], lw=3)
    plt.plot(x, yy, c='k', ls='--', alpha=0.75)
    plt.gca().set_xlim(left=900, right=50000)
    plt.errorbar(uvals, [np.mean(k) for k in uaccs], yerr=[cinterval(k) for k in uaccs], fmt='o', color='k')
    plt.ylabel("Top-1 retrieval accuracy")
    plt.xlabel("Dataset size")

    plt.show()


def dim_size():
    models = base_models + small_models + twod_models
    vals = []
    for it, model in enumerate(models):
        print(it, "model it")
        if model in base_models: 
            vals.append(512)
        elif model in small_models:
            vals.append(32)
        elif model in twod_models:
            vals.append(2)
        else:
            raise ValueError('var doesnt work')

    uvals = sorted(list(set(vals)))
    uaccs = []
    for u in uvals:
        umodels = [model for it, model in enumerate(models) if vals[it] == u]
        uaccs.append(np.array([umodel["final_acc"] for umodel in umodels]))

    plt.gca().set_xscale('log')
    lr = linregress(uvals, [np.mean(k) for k in uaccs])
    x = np.linspace(min(uvals) * 0.5, max(uvals) * 6) 
    y = x * lr.slope + lr.intercept
    a, b = logFit(np.array(uvals), np.array([np.mean(k) for k in uaccs]))
    yy = a + b * np.log(x) 
    plt.plot(uvals, [np.mean(k) for k in uaccs], c=cats[4], lw=3)
    plt.plot(x, yy, c='k', ls='--', alpha=0.75)
    plt.gca().set_xlim(left=1,right=1024)
    plt.errorbar(uvals, [np.mean(k) for k in uaccs], yerr=[cinterval(k) for k in uaccs], fmt='o', color='k')
    plt.ylabel("Top-1 retrieval accuracy")
    plt.xlabel("Number of dimensions in embedding")

    plt.show()
def surv_analysis():
    model = base_models[0]
    mask, length = get_surv([id for it, id in enumerate(model["ids"])])
    
    train_idx = model["test_indicator"] == 0
    test_idx = model["test_indicator"] == 1
    train_df = pd.DataFrame({
        "mask": 1 - mask[train_idx],
        "length": length[train_idx]
    })
    for i in range(model["args"]["repr_dim"]):
        train_df[str(i)] = model["rna-seq"][train_idx][:, i]

    test_df = pd.DataFrame({
        "mask": 1 - mask[test_idx],
        "length": length[test_idx]
    })
    for i in range(model["args"]["repr_dim"]):
        test_df[str(i)] = model["rna-seq"][test_idx][:, i]

    print(train_df)
    good_ids = [it for it in range(np.sum(train_idx)) if not np.isnan(train_df["length"][it]) and not np.isnan(train_df["mask"][it]) and not np.any(np.isnan(train_df[[str(k) for k in range(model["args"]["repr_dim"])]]))]
    train_df = train_df.iloc[good_ids, :]
    good_ids = [it for it in range(np.sum(test_idx)) if not np.isnan(test_df["length"][it]) and not np.isnan(test_df["mask"][it]) and not np.any(np.isnan(test_df[[str(k) for k in range(model["args"]["repr_dim"])]]))]
    test_df = test_df.iloc[good_ids, :]
    
    cph = CoxPHFitter()
    cph.fit(train_df, 'length', 'mask')

    print(cph.score(test_df, scoring_method='concordance_index'))
    pass


def main(fargs):
    for exp_name, model_list in zip([base_exp_name, twod_exp_name, small_exp_name, site_exp_name, data_exp_name], [base_models, twod_models, small_models, site_models, data_models]):
        for file in glob.glob("../../save/" + exp_name + "*/outputs.csv"):
            if file.endswith("0/outputs.csv"):
                continue
            if exp_name not in [try_exp_name, twod_exp_name] and int(file[:file.rfind("/")][-1]) > 1:
                continue
            print(f"reading {file}...")
            
            df = pd.read_csv(file)
            args = file[:file.rfind("/")] + "/0.args"
            with open(args, "r") as f:
                args = json.load(f)

            model = {}

            model["ids"] = df.iloc[:, 0].tolist()
            model["args"] = args
            if site_exp_name == "nosite":
                print(file[:file.rfind("/")].count("_"), "number of underscores")
            if file[:file.rfind("/")].count("_") == 2:
                ext = file[:file.rfind("/")]
                ext = ext[ext.rfind("/") + 1:]
                ext = ext[ext.find("_") + 1:ext.rfind("_")]
                model["var"] = ext
                print(model["var"], "var")
            nl_file = file[:file.rfind("/")] + f"/{epoch}-reports.pth"
            sdict = torch.load(nl_file, map_location=torch.device('cpu'))["state_dict"]
            for key in list(sdict.keys()):
                if key.startswith("module."):
                    sdict[key[7:]] = sdict[key]
                    del sdict[key]
            nl_model = fancy_model.TransformerWithMLP(trns_arch=model["args"]["lm_arch"], out_dim=model['args']['repr_dim'], size=[], batchnorm=True)
            try:
                nl_model.load_state_dict(sdict)
            except:
                nl_model = fancy_model.TransformerWithMLP(trns_arch=model["args"]["lm_arch"], out_dim=model['args']['repr_dim'], size=[], batchnorm=True, old=True)
                nl_model.load_state_dict(sdict)

            model["nl"] = nl_model

            model["test_indicator"] = df["is_test"].to_numpy()

            for t in args["finetune"]:
                if t[-3:] == "_nl":
                    model[t[:-3]] = df[t[:-3]].to_numpy()

                    tdf = pd.read_csv(file[:file.rfind("/") + 1] + f"{epoch}-" + t + ".csv")
                    tidx = [None if id not in tdf[t + "-ids"].tolist() else tdf[t + "-ids"].tolist().index(id) for id in model["ids"]]
                    model[t[:-3] + "_pred"] = np.array([-1 if kk == None else tdf[t + "-pred"].iloc[kk] for kk in tidx])

            for c in args["contrastive"]:
                model[c] = df[[c + "-" + str(i) for i in range(args["repr_dim"])]].to_numpy()

            if "2d" not in exp_name:
                with open(file[:file.rfind("/")] + "/all.log", "r") as f:
                    x = f.read()
                    x = x.split("\n")
                    idx = len(x) - x[-1::-1].index(f"epoch  {epoch} !") - 1

                    if args["lg_types"]:
                        raise NotImplementedError("lg types")
                    
                    shift_num = [it for it, t in enumerate(x[idx:]) if t.startswith("Val loss")]
                    shift_num = shift_num[2] - 1
                    final_acc = float(x[idx + shift_num].split(" ")[-1])

                    shift_num = [it for it, t in enumerate(x[idx:]) if t == "Confusion Matrix:"]
                    shift_num = shift_num[2] + 2
                    num_type = 32 if not args["lg_types"] else -1
                    cm = x[idx + shift_num:idx + shift_num + num_type]
                    types = x[idx + shift_num - 1].split(" ")
                    types = [t for t in types if t != ""]
                    cm = list(map(lambda z: z.split(), cm))
                    cm = list(map(lambda z: z[1:], cm))
                    cm = list(map(lambda z: [y for y in z if y != ""], cm))
                    cm = list(map(lambda z: list(map(float, z)), cm))
                    cm = np.array(cm)

                    shift_num = [it for it, t in enumerate(x[idx:]) if t.startswith("Indices")]
                    shift_num = shift_num[0] + 3
                    class_accs = x[idx + shift_num - 1]
                    class_sizes = x[idx + shift_num]
                    class_rsizes = x[idx + shift_num + 1]

                    class_accs = class_accs.split(" ")
                    class_sizes = class_sizes.split(" ")
                    class_rsizes = class_rsizes.split(" ")
                    class_accs = [float(y) for y in class_accs if y != ""]
                    class_sizes = [float(y) for y in class_sizes if y != ""]
                    class_rsizes = [float(y) for y in class_rsizes if y != ""]

                    model["final_acc"] = final_acc
                    model["type_confusion_matrix"] = cm
                    model["types"] = types
                    model["class_accs"] = class_accs
                    model["real_class_sizes"] = class_sizes
                    model["proj_class_sizes"] = class_rsizes
            else:
                with open(file[:file.rfind("/")] + "/all.log", "r") as f:
                    lines = f.read().split("\n")
                    idx = [it for it, x in enumerate(lines) if "Val acc" in x]
                    l = lines[idx[-2]].split(" ")[-1]
                    model["final_acc"] = float(l)

            if not args["lg_types"]:
                model["num_types"] = 32

            print(f'loaded model from {file}')
            model_list.append(model)

    if fargs.figs == "":
        fargs.figs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    else:
        fargs.figs = [int(x) for x in string_to_list(fargs.figs)]    

    try:
        if 2 in fargs.figs:
            raw_clip_acc()
        if 3 in fargs.figs:
            type_match()
        if 4 in fargs.figs:
            inner_match()
        if 5 in fargs.figs:
            acc_corr()
        if 6 in fargs.figs:
            ensemble(try_models)
        if 7 in fargs.figs:
            twod_plot()
        if 8 in fargs.figs:
            surv_analysis() 
        if 9 in fargs.figs:
            twod_surv_plot()
        if 10 in fargs.figs:
            data_size()
        if 11 in fargs.figs:
            site()
        if 12 in fargs.figs:
            dim_size()
    except Exception:
        print(traceback.format_exc())

    for i in range(0,15):
        subprocess.run("echo $\'\a\'", shell=True)
        time.sleep(3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--figs', default='', type=str)

    args = parser.parse_args()
    main(args)
