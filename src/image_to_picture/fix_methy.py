import pandas as pd
import numpy as np
import pickle
import time


print("Reading Methylation...")
with open("../../data/raw_data/methylation.dat", 'rb') as f:
    methy = pickle.load(f)
    f.close()

methy_mean = methy.groupby('gene1').mean().reset_index()

with open("../../data/raw_data/methylation_mean.dat", 'wb') as ff:
    pickle.dump(methy_mean, ff)
    ff.close()

print("")
