# python imports
import string
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
from math import floor

# sci suite
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ellipj
from scipy import stats
from sklearn.manifold import SpectralEmbedding

# torch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# hugging face
import transformers

verbose = True

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:
             yield item

def type_acronym_mapping(inn):
    mapping_dict = {
        "LAML": "Acute Myeloid Leukemia",
        "ACC": "Adrenocortical carcinoma",
        "BLCA": "Bladder Urothelial Carcinoma",
        "LGG": "Brain Lower Grade Glioma",
        "BRCA": "Breast invasive carcinoma",
        "CESC": "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
        "CHOL": "Cholangiocarcinoma",
        "LCML": "Chronic Myelogenous Leukemia",
        "COAD": "Colon adenocarcinoma",
        "CNTL": "Controls",
        "ESCA": "Esophageal carcinoma",
        "FPPP": "FFPE Pilot Phase II",
        "GBM": "Glioblastoma multiforme",
        "HNSC": "Head and Neck squamous cell carcinoma",
        "KICH": "Kidney Chromophobe",
        "KIRC": "Kidney renal clear cell carcinoma",
        "KIRP": "Kidney renal papillary cell carcinoma",
        "LIHC": "Liver hepatocellular carcinoma",
        "LUAD": "Lung adenocarcinoma",
        "LUSC": "Lung squamous cell carcinoma",
        "DLBC": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
        "MESO": "Mesothelioma",
        "MISC": "Miscellaneous",
        "OV": "Ovarian serous cystadenocarcinoma",
        "PAAD": "Pancreatic adenocarcinoma",
        "PCPG": "Pheochromocytoma and Paraganglioma",
        "PRAD": "Prostate adenocarcinoma",
        "READ": "Rectum adenocarcinoma",
        "SARC": "Sarcoma",
        "SKCM": "Skin Cutaneous Melanoma",
        "STAD": "Stomach adenocarcinoma",
        "TGCT": "Testicular Germ Cell Tumors",
        "THYM": "Thymoma",
        "THCA": "Thyroid carcinoma",
        "UCS": "Uterine Carcinosarcoma",
        "UCEC": "Uterine Corpus Endometrial Carcinoma",
        "UVM": "Uveal Melanoma"
    }

    return [mapping_dict[elem] for elem in inn]

def TCGA_Type_Dataset(args=None):
    df = pd.read_csv("../tcga/huge_clinical_data.tsv", sep='\t', header=0, dtype='str')
    lines_dict = {elem: it for it, elem in enumerate(df["bcr_patient_barcode"])}
    lines = [lines_dict[id] for id in args["ids"]]

    ret = list(df["acronym"].iloc[lines])
    ret_set = sorted(list(set(ret)))
    ret = torch.FloatTensor([ret_set.index(elem) for elem in ret])

    if torch.cuda.is_available():
        ret = ret.cuda() 

    return ret, ret_set

def TCGA_Indicator_Dataset(args=None):
    df = pd.read_csv("../tcga/huge_clinical_data.tsv", sep='\t', header=0, dtype='str')
    lines_dict = {elem: it for it, elem in enumerate(df["bcr_patient_barcode"])}
    lines = [lines_dict[id] for id in args["ids"]]

    if args["prop"] == "lv":
        # lymphovascular invasion for BLCA/TCGT/HNSC
        # size ~750
        # around 330/400 split

        def yn_map(x):
            if x == "YES" and "[" not in x.lower():
                return 1
            elif x == "NO" and "[" not in x.lower():
                return 0
            else:
                return -1

        ret = list(df["lymphovascular_invasion_present"].iloc[lines])
    elif args["prop"] == "perineural":
        # perineural invasion
        # size: 636

        def yn_map(x):
            if x == "YES" and "[" not in x.lower():
                return 1
            elif x == "NO" and "[" not in x.lower():
                return 0
            else:
                return -1

        ret = list(df["perineural_invasion_present"].iloc[lines])
    elif args["prop"] == "papillary":
        # papillary/non papillary for BLCA
        # size 407
        # num positives: 133

        def yn_map(x):
            if x == 'Papillary':
                return 1
            elif x == 'Non-Papillary':
                return 0
            else:
                return -1

        ret = list(df["diagnosis_subtype"].iloc[lines])
    elif args["prop"] == "metastasis":
        # metastasis across several types of cancers
        # size 525
        # pos: 274 vs 261

        def yn_map(x):
           if isinstance(x, str) and "Metastasi" in x:
               return 1
           elif str(x) != "nan":
               return 0
           else:
               return -1

        ret = list(df["new_neoplasm_event_type"].iloc[lines])

    elif args["prop"] == "barretts":
        # baretts esophagus
        # size 396

        def yn_map(x):
            if "no" in x.lower() and "[" not in x.lower():
                return 0
            if "yes" in x.lower() and "[" not in x.lower():
                return 1
            else:
                return -1

        ret = list(df["barretts_esophagus"].iloc[lines])
    elif args["prop"] == "necrosis":
        def yn_map(x):
            if isinstance(x, str) and "absent" in x.lower() and "[" not in x.lower():
                return 0
            elif isinstance(x, str) and "present" in x.lower() and "[" not in x.lower():
                return 1
            else:
                return -1

        ret = list(df["necrosis"].iloc[lines])
    elif args["prop"] == "ulceration":
        def yn_map(x):
            if isinstance(x, str) and "no" in x.lower() and "[" not in x.lower():
                return 0
            elif isinstance(x, str) and "yes" in x.lower() and "[" not in x.lower():
                return 1
            else:
                return -1

        ret = list(df["melanoma_ulceration_indicator"].iloc[lines])
    elif args["prop"] == "mitosis":
        def yn_map(x):
            if isinstance(x, str) and "absent" in x.lower() and "[" not in x.lower():
                return 0
            elif isinstance(x, str) and "present" in x.lower() and "[" not in x.lower():
                return 1
            else:
                return -1

        ret = list(df["mitotic_rate"].iloc[lines])
    # height (2730), margin_status (1709), targeted_molecular_therapy (1675), lymphatic_invasion (790), perineural_invasion_present (636), venous_invasion (710), city_of_procurement (546), headache_history (445), barretts_esophagus (396), eczema_history (356), h_pylori_infection (263), "maximum_tumor_dimension": 208, "goblet_cells_present": 20,
    #  "radiation_therapy": 9618,"pathologic_stage": 6997,"residual_tumor": 4435
    
    ret = list(map(yn_map, ret))

    ret = torch.FloatTensor(ret)
    if torch.cuda.is_available():
        ret = ret.cuda()
    
    return ret

def TCGA_NLIndicator_Dataset(args=None):
    df = pd.read_csv("../tcga/huge_clinical_data.tsv", sep='\t', header=0, dtype='str')
    lines_dict = {elem: it for it, elem in enumerate(df["bcr_patient_barcode"])}
    lines = [lines_dict[id] for id in args["ids"]]

    if args["prop"] == "lv_nl":
        # lymphovascular invasion for BLCA/TCGT/HNSC
        # size ~750
        # around 330/400 split

        def pos(ctype):
            return f"Diagnosis {ctype} . Evidence of lymphovascular invasion"

        def neg(ctype):
            return f"Diagnosis {ctype} ."

        ret = list(df["lymphovascular_invasion_present"].iloc[lines])
    elif args["prop"] == "perineural_nl":
        # perineural invasion
        # size: 636

        def pos(ctype):
            return f"Diagnosis {ctype} . Evidence of perineural invasion"

        def neg(ctype):
            return f"Diagnosis {ctype} ."

        ret = list(df["perineural_invasion_present"].iloc[lines])
    elif args["prop"] == "papillary_nl":
        def pos(ctype):
            return f"Diagnosis papillary {ctype} ."

        def neg(ctype):
            return f"Diagnosis {ctype} ."

        ret = list(df["diagnosis_subtype"].iloc[lines])
    elif args["prop"] == "metastasis_nl":
        # metastasis across several types of cancers
        # size 525
        # pos: 274 vs 261

        def pos(ctype):
            return f"Diagnosis {ctype} . Evidence of distant metastasis"

        def neg(ctype):
            return f"Diagnosis {ctype} ."

        ret = list(df["new_neoplasm_event_type"].iloc[lines])
    elif args["prop"] == "necrosis_nl":
        def pos(ctype):
            return f"Diagnosis {ctype} . Evidence of necrosis"

        def neg(ctype):
            return f"Diagnosis {ctype} ."

        ret = list(df["necrosis"].iloc[lines])
    elif args["prop"] == "ulceration_nl":
        def pos(ctype):
            return f"Diagnosis {ctype} . Evidence of ulceration"
        
        def neg(ctype):
            return f"Diagnosis {ctype} ."

        ret = list(df["melanoma_ulceration_indicator"].iloc[lines])
    elif args["prop"] == "mitosis_nl":
        def pos(ctype):
            return f"Diagnosis {ctype}.  Marked high mitotic activity"
        
        def neg(ctype):
            return f"Diagnosis {ctype} ."

        ret = list(df["mitotic_rate"].iloc[lines])
    elif args["prop"] == "barretts_nl":
        raise ValueError("not implemented barretss yet")
        # baretts esophagus
        # size 396

        def yn_map(inn):
            x = inn[0]
            type = inn[1]
            if "no" in x.lower():
                return f"A {type} not originating from Barrett esophagus" 
            if "yes" in x.lower():
                return f"A {type} originating from Barrett esophagus"
            else:
                return "None" 
            

        ret = list(df["barretts_esophagus"].iloc[lines])
    # height (2730), margin_status (1709), targeted_molecular_therapy (1675), lymphatic_invasion (790), perineural_invasion_present (636), venous_invasion (710), city_of_procurement (546), headache_history (445), barretts_esophagus (396), eczema_history (356), h_pylori_infection (263), "maximum_tumor_dimension": 208, "goblet_cells_present": 20,
    #  "radiation_therapy": 9618,"pathologic_stage": 6997,"residual_tumor": 4435
    def okay_key(s):
        if str(s) == "nan":
            return False
        if "[" in s:
            return False
        if len(str(s)) == 0:
            return False
        if "papillary" in s.lower():
            return True
        if "absent" in s.lower() or "present" in s.lower():
            return True
        if "m1" or "m0" in s.lower():
            return True
        if "no" in s.lower() or "yes" in s.lower():
            return True
        return True
    
    types = TCGA_Type_Dataset(args=args)
    types = type_acronym_mapping([types[1][x] for x in types[0].long().tolist()])
    types = [x if okay_key(ret[it]) else None for it, x in enumerate(types)]

    positives = ["None" if t == None else pos(t) for t in types]
    negatives = ["None" if t == None else neg(t) for t in types]

    if args["lm_arch"] == "distilbert":
        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    elif args["lm_arch"] == "biobert":
        tokenizer = transformers.AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    elif args["lm_arch"] == "bert":
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    batch = tokenizer(["None"] + positives + negatives, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
    none_batched = deepcopy(batch["input_ids"][0])

    for it in range(int(batch["input_ids"].size()[0])):
        if torch.equal(batch["input_ids"][it], none_batched):
            batch["input_ids"][it] = torch.full(batch["input_ids"][it].size(), -1)
            batch["attention_mask"][it] = torch.full(batch["input_ids"][it].size(), -1)

    batch["input_ids"] = batch["input_ids"][1:]
    batch["attention_mask"] = batch["attention_mask"][1:]

    if torch.cuda.is_available():
        batch = batch.to('cuda:0')

    ret = {
            "pos_input_ids": batch["input_ids"][:len(positives)],
            "pos_attention_mask": batch["attention_mask"][:len(positives)],
            "neg_input_ids": batch["input_ids"][len(positives):],
            "neg_attention_mask": batch["attention_mask"][len(positives):]
    }
    
    return ret

def TCGA_Clinical_Dataset(args=None):
    try:
        if args["save"] and json.load(open("../tcga/clinical/combined.args", "r")) == args:
           categorical = torch.load("../tcga/clinical/categorical.pt") 
           continuous = torch.load("../tcga/clinical/continuous.pt") 
        else:
            raise Error("args don't match; regenerating dataset")
    except:
        with open("../tcga/clinical/processed_ids.txt", "r") as f:
            orig_ids = json.load(f)

        data = np.load("../tcga/clinical/processed_clinical_data.npy")
        data = np.transpose(data)
        
        data = data[[orig_ids.index(x) for x in args["ids"]]]

        data = torch.Tensor(data)
        if torch.cuda.is_available():
            data = data.cuda()

        with open("../tcga/clinical/processed_types.txt", "r") as f:
            dtypes = json.load(f)
        
        def dtype_map(x):
            if x in ["numeric"]:
                return "continuous"
            else:
                return "categorical"

        dtypes = list(map(dtype_map, dtypes))

        categorical_idx = [it for it, x in enumerate(dtypes) if x == "categorical"]
        continuous_idx = [it for it, x in enumerate(dtypes) if x == "continuous"]

        categorical = data[:, categorical_idx]
        continuous = data[:, continuous_idx]

        if args["reduce_columns"] != -1:
            for r in range(categorical.size()[1]):
                num_classes = int(torch.max(categorical[:, r]).item() + 1)
                bad_vals = torch.where(torch.bincount(categorical[:, r].long()) < args["reduce_columns"])[0]
                is_bad = torch.Tensor(list(map(lambda x: x in bad_vals, categorical[:, r]))).cuda().bool()
                categorical[is_bad, r] = num_classes
                
                remain_vals = [i for i in range(num_classes + 1) if i not in bad_vals]
                for k in remain_vals:
                    kk = remain_vals.index(k)
                    categorical[categorical[:, r] == k, r] = kk

            categorical = categorical[:, torch.max(categorical, dim=0)[0] >= 1].long()

            continuous = continuous[:, torch.bincount(torch.where(continuous != 0)[1].long()) >= args["reduce_columns"]]

        if args["one_hot"]:
             num_classes = torch.max(categorical, dim=0)[0] + 1
             num_classes_tot = int(torch.sum(num_classes).item())
             oh_categorical = torch.full((categorical.size()[0], num_classes_tot), 0).cuda().long()

             cat_shift = [0] * len(num_classes)
             for i in range(1, len(cat_shift)):
                 cat_shift[i] = cat_shift[i - 1] + num_classes[i - 1]
             cat_shift = torch.Tensor(cat_shift).long()
             
             for it, x in enumerate(num_classes):
                 oh_categorical[torch.arange(0, categorical.size()[0]).long(), cat_shift[it] + categorical[:, it]] = 1

             categorical = oh_categorical

        continuous = continuous - torch.mean(continuous, dim=0, keepdim=True)
        continuous = continuous / torch.std(continuous, dim=0, keepdim=True)

        if args["save"]:
            torch.save(categorical, "../tcga/clinical/categorical.pt")
            torch.save(continuous, "../tcga/clinical/continuous.pt")

            with open("../tcga/clinical/combined.args", "w") as f:
                json.dump(args, f)

    if not args["one_hot"]:
        return {
            "categorical": categorical,
            "continuous": continuous
        }
    else:
        return torch.cat((categorical, continuous), dim=1)

def TCGA_RNA_Dataset(args=None): 
    inn = []

    try:
        if args["save"] and json.load(open("../tcga/rna-seq/combined.args")) == args:
            inn = torch.load("../tcga/rna-seq/combined.pt")
        else:
            raise Error("args don't match; regenerating dataset")
    except:
        if args["copy_ds"] == None:
            gene_names = None
            for it, id in enumerate(args["ids"]):
                if it % 100 == 0:
                    if verbose:
                        print(it, len(args["ids"]), end=" | ", sep="/", flush=True) 
                with open("../tcga/rna-seq/" + id + ".txt", "r") as f:
                    if gene_names != None:
                        inn.append(json.load(f)[1])
                    else:
                        ff = json.load(f)
                        gene_names = ff[0]
                        inn.append(ff[1])
            inn = torch.FloatTensor(inn)

            if args["rna_set"] != '':
                with open(args["rna_set"] + ".txt", "r") as f:
                    ok_genes = json.load(f)
                keep_cols = [it for it, x in enumerate(gene_names) if x in ok_genes] 
                inn = inn[:, keep_cols]
            keep_cols = torch.mean((inn.float() == 0).float(), dim=0) < 1 - args["thresh"]
            inn = inn[:, keep_cols]

            print(inn.size(), "inn size")
            high_percentile = torch.zeros(inn.size()[1])

            if torch.cuda.is_available():
                inn = inn.cuda()
                high_percentile = high_percentile.cuda()
                
            bnum = 10
            for i in range(0, bnum):
                begin = int(inn.size(1) / bnum * i)
                end = int(inn.size(1) / bnum * (i + 1))
                high_percentile[begin:end] = torch.quantile(inn[:, begin:end], 0.95, dim=0, keepdim=True)

            inn = torch.minimum(inn, high_percentile)
            inn = inn - torch.mean(inn, dim=0, keepdim=True)
            inn = inn / torch.std(inn, dim=0, keepdim=True)
        else:
            inn = args["copy_ds"][:]["rna-seq"]
            inn = inn[[args["copy_ds"].tcga_ids.index(k) for k in args["ids"]]]

        print(inn.size(), "inn size 2")
        print(torch.any(torch.isnan(inn)))
        
        if args["save"]:
            torch.save(inn,"../tcga/rna-seq/combined.pt")
            with open("../tcga/rna-seq/combined.args", "w") as f:
                json.dump(args, f)

    if torch.cuda.is_available():
        inn = inn.cuda()

    return inn

def TCGA_ToyReports_Dataset(args=None):
    return None

def TCGA_Reports_Dataset(args=None, input_txt=None):
    if args["manual"]:
        rdir = "../tcga/manual-reports/"
    else:
        rdir = "../tcga/reports/"

    try:
        if args["save"] and json.load(open(rdir + "combined.args")) == args:
           batch = torch.load(rdir + "combined.pt") 
        else:
            raise Error("args don't match; regenerating dataset")
    except:
        if input_txt == None:
            inn = []

            for it, id in enumerate(args["ids"]):
                if it % 100 == 0:
                    if verbose:
                        print(it, len(args["ids"]), end=" | ", sep="/", flush=True) 
                with open(rdir + id + ".txt", "r") as f:
                    if args["manual"]:
                        inn.append(f.read())
                    else:
                        inn.append(json.load(f)[2])
        else:
            inn = [input_txt]

        if args["lm_arch"] == "distilbert":
            tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
        elif args["lm_arch"] == "biobert":
            tokenizer = transformers.AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        elif args["lm_arch"] == "bert":
            tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        batch = tokenizer(inn, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
        if input_txt == None and args["save"]:
            torch.save(batch, rdir + "combined.pt")
            with open(rdir + "combined.args", "w") as f:
                json.dump(args, f)

    if torch.cuda.is_available():
        batch.to('cuda:0')

    return batch

def site_shuffle(l, rand=None, num_split=-1):
    def site_generator(z): 
        if num_split != -1:
            int_alpha = [str(x) for x in range(10)] + [str(x) for x in string.ascii_uppercase]
            return str(z[5] + z[6] + str(int_alpha.index(z[11]) % num_split))
        else:
            return str(z[5] + z[6])

    sites = sorted(list(set([site_generator(x) for x in l])))
    rand.shuffle(sites)
    sites = [[x for x in l if site_generator(x) == s] for s in sites]
    for s in sites:
        rand.shuffle(s)
    sites = [x for s in sites for x in s]
    return sites

class TCGADataHandler():
    def __init__(self, contrastive=['rna-seq', 'reports'], zero_shot=[], finetune=[], train_ratio=0.8, ft_train_ratio=0.5, verbose_=True, lg_types=False, rna_thresh=0.5, clin_thresh=50, rna_set='', seed=3019, rand_shuffle=True, lm_arch='distilbert', clin_one_hot=False):
        rand = random.Random(seed)
        self.rand = rand

        global verbose
        verbose = verbose_

        self.contrastive = contrastive
        self.zero_shot = zero_shot
        self.finetune = finetune

        if lg_types:
            dataset = TCGADataset(contrastive + zero_shot + finetune + ["lg-type"], rna_thresh=rna_thresh, clin_thresh=clin_thresh, rna_set=rna_set, lm_arch=lm_arch, clin_one_hot=clin_one_hot)
        else:
            dataset = TCGADataset(contrastive + zero_shot + finetune, rna_thresh=rna_thresh, clin_thresh=clin_thresh, rna_set=rna_set, lm_arch=lm_arch, clin_one_hot=clin_one_hot)
        
        self._dataset = dataset
        self.nl_type_map = dataset.nl_type_map

        with open("../tcga/data_availability.json", "r") as f:
            avail = json.load(f)

        ids = dataset.tcga_ids

        with open("../tcga/data_availability.json", "r") as f:
            avail = json.load(f)
        m_idx = [id for id in ids if id in avail['manual-reports']]
        czf = [x for x in contrastive if x != "clinical"]
        m_dataset = TCGADataset(czf + ['manual-reports'], save=False, copy_ds=self._dataset, rna_thresh=rna_thresh, rna_set=rna_set, lm_arch=lm_arch, clin_one_hot=clin_one_hot)
        m_types = m_dataset[:]["type"]

        self.manual_dataset = {}
        manual_all = []
        for x in torch.unique(m_types).long().tolist():
            idx = [it for it, t in enumerate(m_types.tolist()) if t == x]
            self.manual_dataset[m_dataset.nl_type_map[x].lower()] = torch.utils.data.Subset(m_dataset, idx)
            manual_all = manual_all + idx
        ids = [id for id in ids if id not in m_dataset.tcga_ids]

        z_tt_idx = [[id for id in ids] for zs in zero_shot] 
        for it, l in enumerate(z_tt_idx):
            z_tt_idx[it] = site_shuffle(l, rand)
        z_train_idx = [l[:floor(len(l) * ft_train_ratio)] for l in z_tt_idx]
        z_test_idx = [l[floor(len(l) * ft_train_ratio):] for l in z_tt_idx]

        if len(z_train_idx) > 0:
            c_tt_idx = [id for id in ids if not all([id not in l for l in z_tt_idx])]
        else:
            c_tt_idx = [id for id in ids]
        if not rand_shuffle:
            c_tt_idx = site_shuffle(c_tt_idx, rand)
        else:
            random.shuffle(c_tt_idx)
        c_pretrain_idx = c_tt_idx[:floor(len(c_tt_idx) * train_ratio)] 
        c_test_idx = c_tt_idx[floor(len(c_tt_idx) * train_ratio):] 

        f_train_idx = c_pretrain_idx + sorted(list(set([id for ztidx in z_train_idx for id in ztidx])))
        f_test_idx = c_test_idx + sorted(list(set([id for ztidx in z_test_idx for id in ztidx])))

        def is_not_n1(obj):
            if torch.is_tensor(obj) and torch.unique(obj).tolist() == [-1]:
                return False
            elif (not torch.is_tensor(obj)) and obj == -1:
                return False
            elif not isinstance(obj, dict):
                return True
            return True

        if len(zero_shot) > 0:
            raise NotImplementedError("no zero shot takss please")
        train_ok = dataset[f_train_idx]
        train_ok = [train_ok[obj] for obj in finetune]
        train_ok = [x if not (isinstance(x, dict) and "pos_input_ids" in x.keys()) else x["pos_input_ids"] for x in train_ok]
        train_ok = [list(map(is_not_n1, t)) for t in train_ok]
        test_ok = dataset[f_test_idx]
        test_ok = [test_ok[obj] for obj in finetune]
        test_ok = [x if not (isinstance(x, dict) and "pos_input_ids" in x.keys()) else x["pos_input_ids"] for x in test_ok]
        test_ok = [list(map(is_not_n1, t)) for t in test_ok]

        f_train_idx = [[x for iit, x in enumerate(f_train_idx) if train_ok[oit][iit]] for oit, obj in enumerate(finetune)]
        f_test_idx = [[x for iit, x in enumerate(f_test_idx) if test_ok[oit][iit]] for oit, obj in enumerate(finetune)]

        self._c_pretrain_idx = [self._dataset.tcga_ids.index(id) for id in c_pretrain_idx]
        self._c_test_idx = [self._dataset.tcga_ids.index(id) for id in c_test_idx]
        self._c_pretrain_ids = [id for id in c_pretrain_idx]
        self._c_test_ids = [id for id in c_test_idx]

        self.pretrain = torch.utils.data.Subset(dataset, c_pretrain_idx)
        self.clip_test = torch.utils.data.Subset(dataset, c_test_idx)
        temp = [y[5:7] for y in c_test_idx]

        self.val_train = {}
        self.val_test = {}
        self.val_train_mini = {}
        for it, z in enumerate(zero_shot):
            self.val_train[z] = torch.utils.data.Subset(dataset, z_train_idx[it])
            self.val_test[z] = torch.utils.data.Subset(dataset, z_test_idx[it])
            self.val_train_mini[z] = torch.utils.data.Subset(dataset, z_train_idx[it][::10])
        for it, f in enumerate(finetune):
            self.val_train[f] = torch.utils.data.Subset(dataset, f_train_idx[it])
            self.val_test[f] = torch.utils.data.Subset(dataset, f_test_idx[it])
            self.val_train_mini[f] = torch.utils.data.Subset(dataset, f_train_idx[it][::10])

        self.num_types = int(torch.max(self._dataset[:]["type"]).item() + 1)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')

        self.nl_prompts = {
                "lv_nl": ["lymphovascular invasion absent", "lymphovascular invasion present"],
                "perineural_nl": ["perineural invasion absent", "perineural invasion present"],
                "metastasis_nl": ["no metastasis", "evidence of metastasis"],
                "papillary_nl": ["non papillary type", "papillary type"],
                "barretts_nl": ["no barrett esophagus", "barett esophagus"]
        }
        for k in list(self.nl_prompts.keys()):
            self.nl_prompts[k] = self.tokenizer(self.nl_prompts[k], padding=True, truncation=True, return_tensors="pt")
            self.nl_prompts[k].to('cuda:0')

    def site_loader(self, dataset, batch_size, drop_last=True):
        if dataset == "pretrain":
            ids = self._c_pretrain_ids
        elif dataset == "test":
            ids = self._c_test_ids
        ids = site_shuffle(ids, self.rand)
        
        return torch.utils.data.DataLoader(
                dataset=torch.utils.data.Subset(self._dataset, ids),
                shuffle=False,
                batch_size=batch_size,
                drop_last=drop_last,
                pin_memory=False,
                num_workers=0
        )

    def by_type(self, batch_size, select_size=-1, dataset="pretrain", reps=1):
        for _ in range(reps):
            if select_size == -1:
                types = range(self.num_types)
            else:
                types = list(range(self.num_types))
                random.shuffle(types)
                types = types[:select_size]
            
            for i in types:
                type_idx = torch.arange(0, len(self._dataset))[self._dataset[:]["type"] == i]
                shuffle_idx = torch.randperm(torch.numel(type_idx))
                type_idx = type_idx[shuffle_idx]
                if dataset == "pretrain":
                    type_idx = np.intersect1d(type_idx, self._c_pretrain_idx)
                elif dataset == "test":
                    type_idx = np.intersect1d(type_idx, self._c_test_idx)
                else:
                    raise ValueError("dataset not recognized")
                type_idx = type_idx[:batch_size]
                type_idx = [self._dataset.tcga_ids[it] for it in type_idx]
                if dataset == "test":
                    assert(all([x in self._c_test_ids for x in type_idx]))
                else:
                    print("ur not in test mode for the bytype loader")

                dl = torch.utils.data.DataLoader(torch.utils.data.Subset(self._dataset, type_idx), batch_size=batch_size)
                for elem in dl:
                    yield elem
    
    def manual_loader(self, batch_size, reps=1, type=''):
        for _ in range(reps):
            dl = torch.utils.data.DataLoader(self.manual_dataset[type], shuffle=True, drop_last=True, batch_size=batch_size)
            for it, elem in enumerate(dl):
                if it > 0:
                    break
                yield elem

    def mask(self, data, ratio=0.2):
        if isinstance(data, dict) and list(sorted(data.keys())) == sorted(["input_ids", "attention_mask"]):
           idx = np.indices(list(data["input_ids"].size()))[:, :, 1:].reshape((2, -1))
           idx = torch.Tensor(idx)
           idx = torch.transpose(idx, 0, 1)
           num_batch = idx.size()[0]
           idx = idx[torch.randperm(num_batch)[:int(num_batch * ratio)], :]
           idx = idx.long()
           data["input_ids"][idx[:, 0], idx[:, 1]] = 103
           return data
        elif torch.is_tensor(data):
           idx = np.indices(list(data.size()))[:, :, 1:].reshape((2, -1))
           idx = torch.Tensor(idx)
           idx = torch.transpose(idx, 0, 1)
           num_batch = idx.size()[0]
           idx = idx[torch.randperm(num_batch)[:int(num_batch * ratio)], :]
           idx = idx.long()
           data[idx[:, 0], idx[:, 1]] = 0
           return data
        elif isinstance(data, dict) and list(sorted(data.keys())) == sorted(["continuous", "categorical"]):
           idx = np.indices(list(data["categorical"].size()))[:, :, 1:].reshape((2, -1))
           idx = torch.Tensor(idx)
           idx = torch.transpose(idx, 0, 1)
           num_batch = idx.size()[0]
           idx = idx[torch.randperm(num_batch)[:int(num_batch * ratio)], :]
           idx = idx.long()
           data["categorical"][idx[:, 0], idx[:, 1]] = 0

           idx = np.indices(list(data["continuous"].size()))[:, :, 1:].reshape((2, -1))
           idx = torch.Tensor(idx)
           idx = torch.transpose(idx, 0, 1)
           num_batch = idx.size()[0]
           idx = idx[torch.randperm(num_batch)[:int(num_batch * ratio)], :]
           idx = idx.long()
           data["continuous"][idx[:, 0], idx[:, 1]] = 0
           return data
        elif isinstance(data, list) and not isinstance(data[0], tuple):
           for i, l in enumerate(data):
               words = l.split(" ")
               idx = list(range(len(words)))
               random.shuffle(idx)
               idx = idx[:int(len(words) * ratio)]
               words = [w if it not in idx else "[PAD]" for it, w in enumerate(words)] 
               data[i] = " ".join(words)
           return data
        elif isinstance(data, list) and isinstance(data[0], tuple):
            return data
        else:
           raise ValueError("mask is not working")

    def mixup(self, dataset, scale=0.2):
        if scale == 0:
            return None, None, dataset
        
        mods = list(dataset.keys())
        size = dataset[mods[0]]
        if isinstance(size, dict):
            size = size[list(size.keys())[0]]
        size = size.size()[0]

        perm_idx = torch.randperm(size)
        lam = torch.unsqueeze(torch.Tensor(np.random.exponential(scale=scale, size=size)), dim=1)
        lam = torch.minimum(lam, torch.full(lam.size(), 1.0)).cuda()
        for m in ['rna-seq']:
            if isinstance(dataset[m], dict):
                if list(sorted(dataset[m].keys())) == sorted(["input_ids", "attention_mask"]):
                    do_mixup = torch.rand(dataset[m]["input_ids"].size()).cuda() < lam
                    dataset[m]["input_ids"][do_mixup] = dataset[m]["input_ids"][perm_idx, :][do_mixup]
                    dataset[m]["input_ids"] = dataset[m]["input_ids"] * dataset[m]["attention_mask"]
                elif list(sorted(dataset[m].keys())) == sorted(["continuous", "categorical"]):
                    do_mixup = torch.rand(dataset[m]["categorical"].size()).cuda() < lam
                    dataset[m]["categorical"][do_mixup] = dataset[m]["categorical"][perm_idx, :][do_mixup]
                    do_mixup = torch.rand(dataset[m]["continuous"].size()).cuda() < lam
                    dataset[m]["continuous"] = dataset[m]["continuous"] * (1 - lam) + dataset[m]["continuous"][perm_idx, :] * lam
                elif torch.is_tensor(dataset[m]):
                    dataset[m] = dataset[m] * (1 - lam) + dataset[m][perm_idx, :] * lam

        return perm_idx.cuda(), lam, dataset

class SiteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, num_matches=2):
        self.ids = dataset[:]["id"]
        self.num_matches = num_matches

    def __iter__(self):
        shuffle_ids = site_shuffle(self.ids, random.Random())
        ret = [self.ids.index(x) for x in shuffle_ids]
        ret = ret[:self.num_matches * floor(len(ret) / self.num_matches)]
        ret = np.reshape(np.array(ret), (-1, self.num_matches))
        np.random.shuffle(ret)
        ret = list(np.reshape(ret, (-1, )))
        return iter(ret)

    def __len__(self):
        return len(self.ids)

# A universal TCGA dataset generator.
class TCGADataset(torch.utils.data.Dataset):
    def __init__(self, data_src, save=True, copy_ds=None, rna_thresh=0.5, clin_thresh=50, rna_set='', lm_arch='distilbert', clin_one_hot=False): 
        self.data_src = data_src

        with open("../tcga/data_availability.json", "r") as f:
            avail = json.load(f)
        good_ids = avail[data_src[0]]
        try:
            for it in range(1, len(data_src)):
                good_ids = [id for id in good_ids if id in avail[data_src[it]]]
        except:
            raise ValueError("not able to find data availabilities; try editing/rerunning availability_script")
        self.tcga_ids = good_ids

        def dataset_map(name):
            if name == "rna-seq": 
                if copy_ds == None:
                    return TCGA_RNA_Dataset(args={"ids": good_ids, "copy_ds": None, "thresh": rna_thresh, "save": save, "rna_set": rna_set})
                else:
                    return TCGA_RNA_Dataset(args={"ids": good_ids, "copy_ds": copy_ds, "thresh": rna_thresh, "save": save, "rna_set": rna_set})
            elif name == "clinical":
                return TCGA_Clinical_Dataset(args={"ids": good_ids, "reduce_columns": clin_thresh, "save": save, "one_hot": clin_one_hot}) 
            elif name == "reports" or name == "clean-reports":
                return TCGA_Reports_Dataset(args={"ids": good_ids, "manual": False, "save": save, "lm_arch": lm_arch})
            elif name == "manual-reports":
                return TCGA_Reports_Dataset(args={"ids": good_ids, "manual": True, "save": save, "lm_arch": lm_arch})
            elif name == "toy_reports":
                return TCGA_ToyReports_Dataset(args={"ids": good_ids, "save": save})
            elif name in ["type", "lg-type"]:
                elem = TCGA_Type_Dataset(args={"ids": good_ids, "save": save})
                self.nl_type_map = elem[1]
                return elem[0]
            elif name in ["lv", "metastasis", "papillary", "perineural", "barretts", "necrosis", "ulceration", "mitosis"]:
                return TCGA_Indicator_Dataset(args={"ids": good_ids, "prop": name, "save": save})
            elif name in ["lv_nl", "metastasis_nl", "papillary_nl", "perineural_nl", "barretts_nl", "necrosis_nl", "ulceration_nl", "mitosis_nl"]:
                return TCGA_NLIndicator_Dataset(args={"ids": good_ids, "prop": name, "save": save, "lm_arch": lm_arch})
            else:
                raise RuntimeError("given data list for TCGA dataset generation is unrecognized")

        self.data = {}
        for src in data_src:
            self.data[src] = dataset_map(src)
        if "type" not in data_src:
            self.data["type"] = dataset_map("type")

    def __getitem__(self, idx):
        if isinstance(idx, str) and idx[:4] == "TCGA":
            return self[self.tcga_ids.index(idx)]
        elif isinstance(idx, list) and all(isinstance(l, str) for l in idx) and all(l[:4] == "TCGA" for l in idx):
            return self[[self.tcga_ids.index(l) for l in idx]]
        else:
            ret = {}
            for src in self.data_src + ["type", "id"]:
                if src in ["id"]:
                    tidx = torch.arange(0, len(self))[idx].tolist()
                    if isinstance(tidx, int):
                        tidx = [tidx]
                    ret[src] = [id for it, id in enumerate(self.tcga_ids) if it in tidx]
                elif torch.is_tensor(self.data[src]):
                    ret[src] = self.data[src][idx]
                elif isinstance(self.data[src], list):
                    tidx = torch.arange(0, len(self))[idx].tolist()
                    if isinstance(tidx, int):
                        tidx = [tidx]
                    ret[src] = [id for it, id in enumerate(self.data[src]) if it in tidx]
                elif isinstance(self.data[src], dict) or "reports" in src or "_nl" in src:
                    small_ret = {}
                    for key in self.data[src].keys():
                        small_ret[key] = self.data[src][key][idx]
                    ret[src] = small_ret
        return ret

    def __len__(self):
        return len(self.tcga_ids)
