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
    df = pd.read_csv("../../tcga/new/huge_clinical_data.tsv", sep='\t', header=0, dtype='str')
    lines_dict = {elem: it for it, elem in enumerate(df["bcr_patient_barcode"])}
    lines = [lines_dict[id] for id in args["ids"]]

    ret = list(df["acronym"].iloc[lines])
    ret_set = sorted(list(set(ret)))
    ret = torch.FloatTensor([ret_set.index(elem) for elem in ret])

    if torch.cuda.is_available():
        ret = ret.cuda() 

    return ret, ret_set

def TCGA_Indicator_Dataset(args=None):
    df = pd.read_csv("../../tcga/new/huge_clinical_data.tsv", sep='\t', header=0, dtype='str')
    lines_dict = {elem: it for it, elem in enumerate(df["bcr_patient_barcode"])}
    lines = [lines_dict[id] for id in args["ids"]]

    if args["prop"] == "lv":
        # lymphovascular invasion for BLCA/TCGT/HNSC
        # size ~750
        # around 330/400 split

        def yn_map(x):
            if x == "YES":
                return 1
            elif x == "NO":
                return 0
            else:
                return -1

        ret = list(df["lymphovascular_invasion_present"].iloc[lines])
    elif args["prop"] == "perineural":
        # perineural invasion
        # size: 636

        def yn_map(x):
            if x == "YES":
                return 1
            elif x == "NO":
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
            if "no" in x.lower():
                return 0
            if "yes" in x.lower():
                return 1
            else:
                return -1

        ret = list(df["barretts_esophagus"].iloc[lines])
    # height (2730), margin_status (1709), targeted_molecular_therapy (1675), lymphatic_invasion (790), perineural_invasion_present (636), venous_invasion (710), city_of_procurement (546), headache_history (445), barretts_esophagus (396), eczema_history (356), h_pylori_infection (263), "maximum_tumor_dimension": 208, "goblet_cells_present": 20,
    #  "radiation_therapy": 9618,"pathologic_stage": 6997,"residual_tumor": 4435
    
    ret = list(map(yn_map, ret))

    ret = torch.FloatTensor(ret)
    if torch.cuda.is_available():
        ret = ret.cuda()
    
    return ret

def TCGA_NLIndicator_Dataset(args=None):
    df = pd.read_csv("../../tcga/new/huge_clinical_data.tsv", sep='\t', header=0, dtype='str')
    lines_dict = {elem: it for it, elem in enumerate(df["bcr_patient_barcode"])}
    lines = [lines_dict[id] for id in args["ids"]]

    if args["prop"] == "lv_nl":
        # lymphovascular invasion for BLCA/TCGT/HNSC
        # size ~750
        # around 330/400 split

        def yn_map(inn):
            x = inn[0]
            type = inn[1]
            if x == "YES":
                return f"A {type} with evidence of lymphovascular invasion"
            elif x == "NO":
                return f"A {type} with no evidence of lymphovascular invasion"
            else:
                return "None"

        ret = list(df["lymphovascular_invasion_present"].iloc[lines])
    elif args["prop"] == "perineural_nl":
        # perineural invasion
        # size: 636

        def yn_map(inn):
            x = inn[0]
            type = inn[1]
            if x == "YES":
                return f"A {type} with evidence of perineural invasion"
            elif x == "NO":
                return f"A {type} with no evidence of perineural invasion"
            else:
                return "None"

        ret = list(df["perineural_invasion_present"].iloc[lines])
    elif args["prop"] == "papillary_nl":
        # papillary/non papillary for BLCA
        # size 407
        # num positives: 133

        def yn_map(inn):
            x = inn[0]
            type = inn[1]
            if x == 'Papillary':
                return f"A papillary {type}"
            elif x == 'Non-Papillary':
                return f"A {type} not papillary"
            else:
                return "None"

        ret = list(df["diagnosis_subtype"].iloc[lines])
    elif args["prop"] == "metastasis_nl":
        # metastasis across several types of cancers
        # size 525
        # pos: 274 vs 261

        def yn_map(inn):
            x = inn[0]
            type = inn[1]
            if str(x) != "nan" and "Metastasi" in x:
                return f"A {type} with evidence of metastasis"
            elif str(x) != "nan":
                return f"A {type} with no evidence of metastasis"
            else:
                return "None"

        ret = list(df["new_neoplasm_event_type"].iloc[lines])
    elif args["prop"] == "barretts_nl":
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
    
    types = TCGA_Type_Dataset(args=args)
    types = type_acronym_mapping([types[1][x] for x in types[0].long().tolist()])
    ret = list(map(yn_map, zip(ret, types)))
    ret = ["None"] + ret

    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    batch = tokenizer(ret, padding=True, truncation=True, return_tensors="pt")
    none_batched = deepcopy(batch["input_ids"][0])

    for it in range(int(batch["input_ids"].size()[0])):
        if torch.equal(batch["input_ids"][it], none_batched):
            batch["input_ids"][it] = torch.full(batch["input_ids"][it].size(), -1)
            batch["attention_mask"][it] = torch.full(batch["input_ids"][it].size(), -1)

    batch["input_ids"] = batch["input_ids"][1:]
    batch["attention_mask"] = batch["attention_mask"][1:]
    if torch.cuda.is_available():
        batch = batch.to('cuda:0')
    
    return batch

def TCGA_Clinical_Dataset(args=None):
    try:
        if args["save"] and json.load(open("../../tcga/new/clinical/combined.args", "r")) == args:
           categorical = torch.load("../../tcga/new/clinical/categorical.pt") 
           continuous = torch.load("../../tcga/new/clinical/continuous.pt") 
        else:
            raise Error("args don't match; regenerating dataset")
    except:
        with open("../../tcga/new/clinical/processed_ids.txt", "r") as f:
            orig_ids = json.load(f)
        # column names cna be accessed by processed_cols

        data = np.load("../../tcga/new/clinical/processed_clinical_data.npy")
        data = np.transpose(data)
        
        data = data[[orig_ids.index(x) for x in args["ids"]]]

        data = torch.Tensor(data)
        if torch.cuda.is_available():
            data = data.cuda()

        with open("../../tcga/new/clinical/processed_types.txt", "r") as f:
            dtypes = json.load(f)
        
        def dtype_map(x):
            if x in ["numeric"]:
                return "continuous"
            else:
                return "categorical"

        #remove clinical indicator columns
        #prop_columns = ["lymphovascular_invasion_present", "perineural_invasion_present", "diagnosis_subtype", "new_neoplasm_event_type", "barretts_esophagus"]
        #bad_idx = [dtypes.index(col) for col in prop_columns]
        #good_idx = [idx for idx in range(len(dtypes)) if idx not in bad_idx]
        #data = data[good_idx]
        dtypes = list(map(dtype_map, dtypes))
        #dtypes = dtypes[good_idx]
        #assert(np.size(data, axis=0) > np.size(data, axis=1))

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

            categorical = categorical[:, torch.max(categorical, dim=0)[0] >= 1]

            continuous = continuous[:, torch.bincount(torch.where(continuous != 0)[1].long()) >= args["reduce_columns"]]

        continuous = continuous - torch.mean(continuous, dim=0, keepdim=True)
        continuous = continuous / torch.std(continuous, dim=0, keepdim=True)

        if args["save"]:
            torch.save(categorical, "../../tcga/new/clinical/categorical.pt")
            torch.save(continuous, "../../tcga/new/clinical/continuous.pt")

            with open("../../tcga/new/clinical/combined.args", "w") as f:
                json.dump(args, f)

    return {
        "categorical": categorical.long(),
        "continuous": continuous
    }

def TCGA_RNA_Dataset(args=None): # TODO: option to generate new data (reports as well!)
    inn = []

    try:
        if args["save"] and json.load(open("../../tcga/new/rna-seq/combined.args")) == args:
            inn = torch.load("../../tcga/new/rna-seq/combined.pt")
        else:
            raise Error("args don't match; regenerating dataset")
    except:
        if args["copy_ds"] == None:
            gene_names = None
            for it, id in enumerate(args["ids"]):
                if it % 100 == 0:
                    if verbose:
                        print(it, len(args["ids"]), end=" | ", sep="/", flush=True) 
                with open("../../tcga/new/rna-seq/" + id +".txt", "r") as f:
                    if gene_names != None:
                        inn.append(json.load(f)[1])
                    else:
                        ff = json.load(f)
                        gene_names = ff[0]
                        inn.append(ff[1])
            inn = torch.FloatTensor(inn)
 
            #if args["rna_set"] != '':
            #    with open(args["rna_set"] + ".txt", "r") as f:
            #        ok_genes = json.load(f)
            #    keep_cols = [it for it, x in enumerate(gene_names) if x in ok_genes] 
            #    inn = inn[:, keep_cols]
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
                print(begin)
                print(end)
                print(i, "i")
                print(inn[:, begin:end])
                high_percentile[begin:end] = torch.quantile(inn[:, begin:end], 0.95, dim=0, keepdim=True)
 
            # normalize
            inn = torch.minimum(inn, high_percentile)
            #inn = torch.log(1 + inn)
            inn = inn - torch.mean(inn, dim=0, keepdim=True)
            inn = inn / torch.std(inn, dim=0, keepdim=True)
        else:
            inn = args["copy_ds"][:]["rna-seq"]
            inn = inn[[args["copy_ds"].tcga_ids.index(k) for k in args["ids"]]]
        
        if args["save"]:
            torch.save(inn,"../../tcga/new/rna-seq/combined.pt")
            with open("../../tcga/new/rna-seq/combined.args", "w") as f:
                json.dump(args, f)

    if torch.cuda.is_available():
        inn = inn.cuda()

    return inn

def TCGA_ToyReports_Dataset(args=None):
    return None

    types, desc = TCGA_Type_Dataset(args=args)
    types = types.tolist()
    types = [desc[int(elem)] for elem in types]
    types = type_acronym_mapping(types)

    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    batch = tokenizer(types, padding=True, truncation=True, return_tensors="pt")

    if torch.cuda.is_available():
        batch.to('cuda:0')

    return batch

def TCGA_Reports_Dataset(args=None, input_txt=None):
    if args["manual"]:
        rdir = "../../tcga/new/manual-reports/"
    else:
        rdir = "../../tcga/new/reports/"

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

        # TODO: ADD MEASUREMENTS & FLOATS SUPPORT
        #text = text.split(" ")
        #def is_measurement(word):
        #    return ((text[-2:] == "cm" or text[-2:] == "mm") and isfloat(text[:-2])) or ((text[-3:-1] == "cm" or text[-3:-1] == "mm") and isfloat(text[-3:-1]))
        #text = [word for word in text if not is_measurement(word)]

        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')

        batch = tokenizer(inn, padding='max_length', max_length=150, truncation=True, return_tensors="pt")
        if input_txt == None and args["save"]:
            torch.save(batch, rdir + "combined.pt")
            with open(rdir + "combined.args", "w") as f:
                json.dump(args, f)

    if torch.cuda.is_available():
        batch.to('cuda:0')

    return batch

def site_shuffle(l, rand=None, num_split=-1):
    rand = random.Random()

    def site_generator(z): # splits each site into 5 parts
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
    def __init__(self, contrastive=['rna-seq', 'reports'], zero_shot=[], finetune=[], train_ratio=0.8, ft_train_ratio=0.5, verbose_=True, lg_types=False):
        global verbose
        verbose = verbose_

        self.contrastive = contrastive
        self.zero_shot = zero_shot
        self.finetune = finetune

        if lg_types:
            dataset = TCGADataset(contrastive + zero_shot + finetune + ["lg-type"])
        else:
            dataset = TCGADataset(contrastive + zero_shot + finetune)
        
        self._dataset = dataset

        with open("../../tcga/new/data_availability.json", "r") as f:
            avail = json.load(f)

        ids = dataset.tcga_ids

        with open("../../tcga/new/data_availability.json", "r") as f:
            avail = json.load(f)
        m_idx = [id for id in ids if id in avail['manual-reports']]
        print("before data generation")
        m_dataset = TCGADataset(contrastive + zero_shot + finetune + ['manual-reports'], save=False, copy_ds=self._dataset)
        print("after data generation")
        m_types = m_dataset[:]["type"]

        self.manual_dataset = {}
        manual_all = []
        for x in torch.unique(m_types).long().tolist():
            idx = [it for it, t in enumerate(m_types.tolist()) if t == x]
            self.manual_dataset[m_dataset.nl_type_map[x].lower()] = torch.utils.data.Subset(m_dataset, idx)
            manual_all = manual_all + idx
        ids = [id for id in ids if id not in m_dataset.tcga_ids]

        z_tt_idx = [[id for id in ids] for zs in zero_shot] 
        for l in z_tt_idx:
            site_shuffle(l)
            print(l[:100])
        z_train_idx = [l[:floor(len(l) * ft_train_ratio)] for l in z_tt_idx]
        z_test_idx = [l[floor(len(l) * ft_train_ratio):] for l in z_tt_idx]

        if len(z_train_idx) > 0:
            c_tt_idx = [id for id in ids if not all([id not in l for l in z_tt_idx])]
        else:
            c_tt_idx = [id for id in ids]
        #site_shuffle(c_tt_idx)
        print(c_tt_idx[:100])
        random.shuffle(c_tt_idx)
        print(c_tt_idx[:100])
        c_pretrain_idx = c_tt_idx[:floor(len(c_tt_idx) * train_ratio)] 
        c_test_idx = c_tt_idx[floor(len(c_tt_idx) * train_ratio):] 

        # if len(z_train_idx) > 0:
        #     random.shuffle(z_train_idx)
        #     f_tt_idx = c_test_idx + z_train_idx[0][floor(len(z_train_idx) * train_ratio):]
        # else:
        #     f_tt_idx = c_test_idx
        # random.shuffle(f_tt_idx)
        # f_train_idx = f_tt_idx[:floor(len(f_tt_idx) * ft_train_ratio)]
        # f_test_idx = f_tt_idx[floor(len(f_tt_idx) * ft_train_ratio):]
        f_train_idx = c_pretrain_idx + list(set([id for ztidx in z_train_idx for id in ztidx]))
        f_test_idx = c_test_idx + list(set([id for ztidx in z_test_idx for id in ztidx]))

        def is_not_n1(obj):
            if obj == -1:
                return False
            elif not isinstance(obj, dict):
                return True
            elif "input_ids" not in obj.keys():
                return True 
            elif torch.unique(obj["input_ids"]).tolist() == [-1]:
                return False
            return True

        z_train_idx = [[x for x in z_train_idx[it] if is_not_n1(dataset[x][obj])] for it, obj in enumerate(zero_shot)]
        z_test_idx = [[x for x in z_test_idx[it] if is_not_n1(dataset[x][obj])] for it, obj in enumerate(zero_shot)]
        f_train_idx = [[x for x in f_train_idx if is_not_n1(dataset[x][obj])] for obj in finetune]
        f_test_idx = [[x for x in f_test_idx if is_not_n1(dataset[x][obj])] for obj in finetune]

        self._c_pretrain_idx = [ids.index(id) for id in c_pretrain_idx]
        self._c_test_idx = [ids.index(id) for id in c_test_idx]
        self.pretrain = torch.utils.data.Subset(dataset, c_pretrain_idx)
        self.clip_test = torch.utils.data.Subset(dataset, c_test_idx)
        print(len(self.pretrain), "pretrain")
        print(len(self.clip_test), "clipests")

        self.val_train = {}
        self.val_test = {}
        self.val_train_mini = {}
        for it, z in enumerate(zero_shot):
            self.val_train[z] = torch.utils.data.Subset(dataset, z_train_idx[it])
            self.val_test[z] = torch.utils.data.Subset(dataset, z_test_idx[it])
            self.val_train_mini[z] = torch.utils.data.Subset(dataset, z_train_idx[it][::10])
            print(len(self.val_train[z]), z)
            print(len(self.val_test[z]), z)
            print(len(self.val_train_mini[z]), z, "mini")
        for it, f in enumerate(finetune):
            self.val_train[f] = torch.utils.data.Subset(dataset, f_train_idx[it])
            self.val_test[f] = torch.utils.data.Subset(dataset, f_test_idx[it])
            self.val_train_mini[f] = torch.utils.data.Subset(dataset, f_train_idx[it][::10])
            print(len(self.val_train[f]), f)
            print(len(self.val_test[f]), f)
            print(len(self.val_train_mini[f]), f, "mini")

        self.num_types = int(torch.max(self._dataset[:]["type"]).item() + 1)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')


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

# A universal TCGA dataset generator.
class TCGADataset(torch.utils.data.Dataset):
    def __init__(self, data_src, save=True, copy_ds=None): # implement these new parameters!! & 3 outputs not one
        self.data_src = data_src

        with open("../../tcga/new/data_availability.json", "r") as f:
            avail = json.load(f)
        good_ids = avail[data_src[0]]
        try:
            for it in range(1, len(data_src)):
                good_ids = [id for id in good_ids if id in avail[data_src[it]]]
        except:
            raise ValueError("not able to find data availabilities; try editing/rerunning availability_script")
        self.tcga_ids = good_ids

        def dataset_map(name):
            if name == "rna-seq": # TODO: add cna,mutations,clinical
                if copy_ds == None:
                    return TCGA_RNA_Dataset(args={"ids": good_ids, "copy_ds": None, "thresh": 0.5, "save": save})
                else:
                    return TCGA_RNA_Dataset(args={"ids": good_ids, "copy_ds": copy_ds, "thresh": 0.5, "save": save})
            elif name == "clinical":
                return TCGA_Clinical_Dataset(args={"ids": good_ids, "reduce_columns": 50, "save": save}) 
            elif name == "reports" or name == "clean-reports":
                return TCGA_Reports_Dataset(args={"ids": good_ids, "manual": False, "save": save})
            elif name == "manual-reports":
                return TCGA_Reports_Dataset(args={"ids": good_ids, "manual": True, "save": save})
            elif name == "toy_reports":
                return TCGA_ToyReports_Dataset(args={"ids": good_ids, "save": save})
            elif name in ["type", "lg-type"]:
                elem = TCGA_Type_Dataset(args={"ids": good_ids, "save": save})
                self.nl_type_map = elem[1]
                return elem[0]
            elif name in ["lv", "metastasis", "papillary", "perineural", "barretts"]:
                return TCGA_Indicator_Dataset(args={"ids": good_ids, "prop": name, "save": save})
            elif name in ["lv_nl", "metastasis_nl", "papillary_nl", "perineural_nl", "barretts_nl"]:
                return TCGA_NLIndicator_Dataset(args={"ids": good_ids, "prop": name, "save": save})
            else:
                raise RuntimeError("given data list for TCGA dataset generation is unrecognized")

        self.data = {}
        for src in data_src:
            self.data[src] = dataset_map(src)
        if "type" not in data_src:
            self.data["type"] = dataset_map("type")

    """def __init__(self, contrastive=['rna-seq', 'reports'], zero_shot=[], split=[verbose_=True):
        global verbose
        verbose = verbose_

        self.lr_data_types = deepcopy(lr_data)
        self.target_type = target

        # First, find the IDs that have all data types.
        with open("../../tcga/new/data_availability.json", "r") as f:
            avail = json.load(f)
        if target == "":
            data_src = lr_data
        else:
            data_src = lr_data + [target]
        good_ids = avail[data_src[0]]
        for it in range(1, len(data_src)):
            good_ids = [id for id in good_ids if id in avail[data_src[it]]]
        self.tcga_ids = good_ids

        # Now, map dataset names to IDs.
        def dataset_map(name): # TODO include subtype data as well & empty
            if name == "rna-seq": # TODO: add cna,mutations,clinical
                return TCGA_RNA_Dataset(args={"ids": good_ids, "thresh": 0.5})
            elif name == "clinical":
                return TCGA_Clinical_Dataset(good_ids) #TODO: implement
            elif name == "reports" or name == "clean-reports":
                return TCGA_Reports_Dataset(args={"ids": good_ids})
            elif name == "toy_reports":
                return TCGA_ToyReports_Dataset(args={"ids": good_ids})
            elif name == "type":
                d = TCGA_Type_Dataset(args={"ids": good_ids})
                return d[0]
            elif name == "coad_subtype":
                return TCGA_Subtypes_Dataset(type='coad') # TOOD: implement
            else:
                raise RuntimeError("given data list for TCGA dataset generation is unrecognized")

        for it, data_type in enumerate(data_src):
            data_src[it] = dataset_map(data_type)

        self.lr = data_src[:len(lr_data)]
        if target != "":
            self.target = data_src[-1]
        else:
            self.target = None
        input(len(self))"""

    def __getitem__(self, idx):
        # def safe_index(data, idx, data_type):
        #     if data_type == "rna-seq":
        #         return data[idx]
        #     elif data_type == "type":
        #         return data[idx]
        #     elif data_type == "clinical":
        #         small_ret = {}
        #         for key in data.keys():
        #             small_ret[key] = data[key][idx]
        #         return small_ret
        #     elif data_type == "reports" or data_type == "clean-reports":
        #         small_ret = {}
        #         for key in data.keys():
        #             small_ret[key] = data[key][idx]
        #         return small_ret
        #     elif data_type == "toy_reports":
        #         small_ret = {}
        #         for key in data.keys():
        #             small_ret[key] = data[key][idx]
        #         return small_ret

        # ret = {}
        # ret["left"] = safe_index(self.lr[0], idx, self.lr_data_types[0])
        # if len(self.lr) > 1:
        #     ret["right"] = safe_index(self.lr[1], idx, self.lr_data_types[1])
        # if self.target != None:
        #     ret["target"] = safe_index(self.target, idx, self.target_type)

        if isinstance(idx, str) and idx[:4] == "TCGA":
            return self[self.tcga_ids.index(idx)]
        elif isinstance(idx, list) and all(isinstance(l, str) for l in idx) and all(l[:4] == "TCGA" for l in idx):
            return self[[self.tcga_ids.index(l) for l in idx]]
        else:
            ret = {}
            for src in self.data_src + ["type"]:
                if src in ["rna-seq", "type", "lv", "metastasis", "papillary", "perineural", "barretts"]:
                    ret[src] = self.data[src][idx]
                elif src in ["clinical", "reports", "clean-reports", "manual-reports", "toy_reports", "lv_nl", "metastasis_nl", "papillary_nl", "perineural_nl", "barretts_nl"]:
                    small_ret = {}
                    for key in self.data[src].keys():
                        small_ret[key] = self.data[src][key][idx]
                    ret[src] = small_ret

        return ret

    def __len__(self):
        # if self.lr_data_types[0] == "rna-seq":
        #     return np.size(self.lr[0], axis=0)
        # elif self.lr_data_types[0] == "type":
        #     return np.size(self.lr[0], axis=0)
        # elif self.lr_data_types[0] in ["reports", "clean-reports"]:
        #     return np.size(self.lr[0]["input_ids"], axis=0)
        # elif self.lr_data_types[0] == "toy_reports":
        #     return np.size(self.lr[0]["input_ids"], axis=0)
        # elif self.lr_data_typse[0] == "clinical":
        #     return np.size(self.lr[0]["data"], axis=0)
        return len(self.tcga_ids)

