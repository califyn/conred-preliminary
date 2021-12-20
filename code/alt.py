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
#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_context("paper")

flare = sns.color_palette("flare", as_cmap=True)
cats = sns.color_palette("Set2")
gold = sns.color_palette("YlOrBr", as_cmap=True)

def auc(): 
    arr = [[.683, .461, .705, .548, .579],
           [.671, .437, .682, .522, .563],
           [.7, .438, .712, .537, .543],
           [.756, .467, .653, .567, .29],
           [.699, .545, .722, .545, .609]]

    arr = np.array(arr).T
    rrr = [0, 2, 3, 4, 1]
    l = ["Papillary", "LVI", "PNI", "Metastasis", "Ulceration"]
    plt.bar([0, 1, 2, 3, 4], [np.mean(arr[it]) for it in rrr], tick_label=[l[k] for k in rrr], color=cats)
    plt.xticks(rotation=20)
    plt.gca().legend(loc='best', fontsize=8)
    plt.ylabel("AUROC")
    plt.axhline(y=0.5, ls='--', c='k')

    plt.show()


auc()
