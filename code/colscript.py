import pandas as pd
import json
from collections import OrderedDict

# view columns of the data

df = pd.read_csv("../tcga/huge_clinical_data.tsv", sep='\t', header=0, dtype='str')
cols = df.columns.tolist()

lcols = []
for c in cols:
    lcols.append(len([x for x in df[c] if str(x) != "nan" and "[" not in x]))
print(lcols)

cols = [x for _, x in sorted(zip(lcols, cols), key=lambda pair: pair[0])]
lcols = list(sorted(lcols))
print(cols)

dcols = OrderedDict()
for it, c in enumerate(cols):
    dcols[c] = lcols[it]

print(json.dumps(dcols, indent=4))
