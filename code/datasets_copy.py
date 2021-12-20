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
            if "Metastasi" in x:
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

        def yn_map(x, type):
            if x == "YES":
                return f"A {type} with evidence of lymphovascular invasion"
            elif x == "NO":
                return f"A {type} with no evidence of lymphovascular invasion"
            else:
                return -1

        ret = list(df["lymphovascular_invasion_present"].iloc[lines])
    elif args["prop"] == "perineural_nl":
        # perineural invasion
        # size: 636

        def yn_map(x, type):
            if x == "YES":
                return f"A {type} with evidence of perineural invasion"
            elif x == "NO":
                return f"A {type} with no evidence of perineural invasion"
            else:
                return -1

        ret = list(df["perineural_invasion_present"].iloc[lines])
    elif args["prop"] == "papillary_nl":
        # papillary/non papillary for BLCA
        # size 407
        # num positives: 133

        def yn_map(x, type):
            if x == 'Papillary':
                return f"A papillary {type}"
            elif x == 'Non-Papillary':
                return f"A {type} not papillary"
            else:
                return -1

        ret = list(df["diagnosis_subtype"].iloc[lines])
    elif args["prop"] == "metastasis_nl":
        # metastasis across several types of cancers
        # size 525
        # pos: 274 vs 261

        def yn_map(x, type):
            if "Metastasi" in x:
                return f"A {type} with evidence of metastasis"
            elif str(x) != "nan":
                return f"A {type} with no evidence of metastasis"
            else:
                return -1

        ret = list(df["new_neoplasm_event_type"].iloc[lines])
    elif args["prop"] == "barretts_nl":
        # baretts esophagus
        # size 396

        def yn_map(x, type):
            if "no" in x.lower():
                return f"A {type} not originating from Barrett esophagus" 
            if "yes" in x.lower():
                return f"A {type} originating from Barrett esophagus"
            else:
                return -1

        ret = list(df["barretts_esophagus"].iloc[lines])
    # height (2730), margin_status (1709), targeted_molecular_therapy (1675), lymphatic_invasion (790), perineural_invasion_present (636), venous_invasion (710), city_of_procurement (546), headache_history (445), barretts_esophagus (396), eczema_history (356), h_pylori_infection (263), "maximum_tumor_dimension": 208, "goblet_cells_present": 20,
    #  "radiation_therapy": 9618,"pathologic_stage": 6997,"residual_tumor": 4435
    
    ret = list(map(yn_map, zip(ret, map(type_acronym_mapping, TCGA_Type_Dataset(args=args).tolist()))))

    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    batch = tokenizer(ret, padding=True, truncation=True, return_tensors="pt")

    if torch.cuda.is_available():
        batch = batch.cuda()
    
    return batch

def TCGA_Clinical_Dataset(args=None):
    with open("../../tcga/new/clinical/processed_ids.txt", "r") as f:
        orig_ids = json.load(f)
    # column names cna be accessed by processed_cols

    data = np.load("../../tcga/new/clinical/processed_clinical_data.npy")
    
    data = data[[orig_ids.index(x) for x in args["ids"]]]

    data = torch.Tensor(data)
    if torch.cuda.is_available():
        data = data.cuda()

    with open("../../tcga/new/clinical/processed_types.txt", "r") as f:
        dtypes = json.load(f)
    
    def dtype_map(x):
        if x in ["numerical"]:
            return "continuous"
        else:
            return "categorical"

    #remove clinical indicator columns
    prop_columns = ["lymphovascular_invasion_present", "perineural_invasion_present", "diagnosis_subtype", "new_neoplasm_event_type", "barretts_esophagus"]
    bad_idx = [dtypes.index(col) for col in prop_columns]
    good_idx = [idx for idx in range(len(dtypes)) if idx not in bad_idx]
    data = data[good_idx]
    assert(np.size(data, axis=0) > np.size(data, axis=1))

    dtypes = list(map(dtype_map, dtypes))

    return {
        "data": data,
        "col_types": dtypes
    }

def TCGA_RNA_Dataset(args=None): # TODO: option to generate new data (reports as well!)
    inn = []

    try:
        if json.load(open("../../tcga/new/rna-seq/combined.args")) == args:
            inn = torch.load("../../tcga/new/rna-seq/combined.pt")
        else:
            raise Error("args don't match; regenerating dataset")
    except:
        for it, id in enumerate(args["ids"]):
            if it % 100 == 0:
                if verbose:
                    print(it, len(args["ids"]), end=" | ", sep="/", flush=True) 
            with open("../../tcga/new/rna-seq/" + id + ".txt", "r") as f:
                inn.append(json.load(f)[1])

            with open("../../tcga/new/rna-seq/" + id + ".txt", "r") as f:
                inn.append(json.load(f)[1])

        inn = torch.FloatTensor(inn)

        inn = inn[:, torch.mean((inn.float() == 0).float(), dim=0) < 1 - args["thresh"]]
        high_percentile = torch.zeros(inn.size()[1])

        if torch.cuda.is_available():
            inn = inn.cuda()
            high_percentile = high_percentile.cuda()
            
        bnum = 10
        for i in range(0, bnum):
            begin = int(inn.size(1) / bnum * i)
            end = int(inn.size(1) / bnum * (i + 1))
            high_percentile[begin:end] = torch.quantile(inn[:, begin:end], 0.95, dim=0, keepdim=True)

        # normalize
        inn = torch.minimum(inn, high_percentile)
        inn = inn - torch.mean(inn, dim=0, keepdim=True)
        inn = inn / torch.std(inn, dim=0, keepdim=True)
        
        torch.save(inn,"../../tcga/new/rna-seq/combined.pt")
        with open("../../tcga/new/rna-seq/combined.args", "w") as f:
            json.dump(args, f)

    if torch.cuda.is_available():
        inn = inn.cuda()

    return inn

def TCGA_ToyReports_Dataset(args=None):
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
    try:
        if json.load(open("../../tcga/new/reports/combined.args")) == args:
            batch = torch.load("../../tcga/new/reports/combined.pt")
        else:
            raise Error("args don't match; regenerating dataset")
    except:
        if input_txt == None:
            inn = []

            for it, id in enumerate(args["ids"]):
                if it % 100 == 0:
                    if verbose:
                        print(it, len(args["ids"]), end=" | ", sep="/", flush=True) 
                with open("../../tcga/new/reports/" + id + ".txt", "r") as f:
                    inn.append(json.load(f)[2])
        else:
            inn = [input_txt]

        # TODO: ADD MEASUREMENTS & FLOATS SUPPORT
        #text = text.split(" ")
        #def is_measurement(word):
        #    return ((text[-2:] == "cm" or text[-2:] == "mm") and isfloat(text[:-2])) or ((text[-3:-1] == "cm" or text[-3:-1] == "mm") and isfloat(text[-3:-1]))
        #text = [word for word in text if not is_measurement(word)]

        tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')

        batch = tokenizer(inn, padding=True, truncation=True, return_tensors="pt")
        if input_txt == None:
            torch.save(batch, "../../tcga/new/reports/combined.pt")
            with open("../../tcga/new/reports/combined.args", "w") as f:
                json.dump(args, f)

    if torch.cuda.is_available():
        batch.to('cuda:0')

    return batch


class TCGADataHandler():
    def __init__(self, contrastive=['rna-seq', 'reports'], zero_shot=[], finetune=[], train_ratio=0.8, verbose_=True):
        global verbose
        verbose = verbose_

        self.contrastive = contrastive
        self.zero_shot = zero_shot
        self.finetune = finetune

        dataset = TCGADataset(contrastive + zero_shot + finetune)
        self._dataset = dataset

        with open("../../tcga/new/data_availability.json", "r") as f:
            avail = json.load(f)

        ids = dataset.tcga_ids

        z_tt_idx = [[id for id in ids] for zs in zero_shot] 
        for l in z_tt_idx:
            random.shuffle(l)
        z_train_idx = [l[:floor(len(l) * train_ratio)] for l in z_tt_idx]
        z_test_idx = [l[floor(len(l) * train_ratio):] for l in z_tt_idx]

        c_tt_idx = [id for id in ids if not all([id not in l for l in z_tt_idx])]
        random.shuffle(l)
        c_pretrain_idx = c_tt_idx[:floor(len(c_tt_idx) * train_ratio)] 
        c_test_idx = c_tt_idx[floor(len(c_tt_idx) * train_ratio):] 

        random.shuffle(z_train_idx)
        f_tt_idx = c_test_idx + z_train_idx[floor(len(z_train_idx) * train_ratio):][0]
        random.shuffle(f_tt_idx)
        f_train_idx = f_tt_idx[:floor(len(f_tt_idx) * train_ratio)]
        f_test_idx = f_tt_idx[floor(len(f_tt_idx) * train_ratio):]

        z_train_idx = [[x for x in z_train_idx[it] if dataset[x][obj] != -1] for it, obj in enumerate(zero_shot)]
        z_test_idx = [[x for x in z_test_idx[it] if dataset[x][obj] != -1] for it, obj in enumerate(zero_shot)]
        f_train_idx = [[x for x in f_train_idx if dataset[x][obj] != -1] for obj in finetune]
        f_test_idx = [[x for x in f_test_idx if dataset[x][obj] != -1] for obj in finetune]

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

        print(self.clip_test[:]["rna-seq"])
        print(self.clip_test[:]["type"])


# A universal TCGA dataset generator.
class TCGADataset(torch.utils.data.Dataset):
    def __init__(self, data_src): # implement these new parameters!! & 3 outputs not one
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
                return TCGA_RNA_Dataset(args={"ids": good_ids, "thresh": 0.5})
            elif name == "clinical":
                return TCGA_Clinical_Dataset(good_ids) 
            elif name == "reports" or name == "clean-reports":
                return TCGA_Reports_Dataset(args={"ids": good_ids})
            elif name == "toy_reports":
                return TCGA_ToyReports_Dataset(args={"ids": good_ids})
            elif name == "type":
                return TCGA_Type_Dataset(args={"ids": good_ids})[0]
            elif name in ["lv", "metastasis", "papillary", "perineural", "barretts"]:
                return TCGA_Indicator_Dataset(args={"ids": good_ids, "prop": name})
            elif name in ["lv_nl", "metastasis_nl", "papillary_nl", "perineural_nl", "barretts_nl"]:
                return TCGA_NLIndicator_Dataset(args={"ids": good_ids, "prop": name})
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
                elif src in ["clinical", "reports", "clean-reports", "toy_reports", "lv_nl", "metastasis_nl", "papillary_nl", "perineural_nl", "barretts_nl"]:
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
