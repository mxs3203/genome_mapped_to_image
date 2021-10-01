import pandas as pd
import numpy as np
import pickle
import time
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
meta = pd.read_csv("../../data/meta_data.csv")
print(meta['type'].value_counts())
meta['encoded_type'] = le.fit_transform(meta['type'])

print(meta['encoded_type'].value_counts())
print(meta)
# print("Reading Methylation...")
# with open("../../data/raw_data/methylation.dat", 'rb') as f:
#     methy = pickle.load(f)
#     f.close()
#
# methy_mean = methy.groupby('gene1').mean().reset_index()
#
# with open("../../data/raw_data/methylation_mean.dat", 'wb') as ff:
#     pickle.dump(methy_mean, ff)
#     ff.close()
#
# print("")
