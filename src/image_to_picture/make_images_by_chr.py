import argparse
import math
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import make_image_chr, find_mutations, find_losses, find_gains, \
    find_gene_expression, find_methylation

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output',type=str, default='TP53_data/SquereImg')
parser.add_argument('--tp53', type=int, default='0')
parser.add_argument('--shuffle', type=int, default='0')

args = parser.parse_args()
args.tp53 = bool(int(args.tp53))
args.shuffle = bool(int(args.shuffle))
folder = args.output #'TP53_data/ShuffleImg'
print(args)

start_time = time.time()
print("Reading clinical...")
clinical = pd.read_csv("../../data/raw_data/clinical_all_cancer_with_300_samples.csv")
print("Reading ascat...")
ascat = pd.read_csv("../../data/raw_data/ascat.csv")
ascat_loss = ascat.loc[ascat['loss'] == True]
ascat_gain = ascat.loc[ascat['gain'] == True]
print("Reading all gene definition...")
all_genes = pd.read_csv("../../data/raw_data/all_genes_ordered_by_chr.csv")
if args.shuffle:
    print("Shuffling gene list")
    all_genes = all_genes.sample(frac=1).reset_index(drop=True) # Shuffle genes
if args.tp53:
    all_genes = all_genes[all_genes['name2'] != "TP53"]
print("Reading Muts...")
muts = pd.read_csv("../../data/raw_data/muts.csv")
print("Reading gene exp...")
gene_exp = pd.read_csv("../../data/raw_data/gene_exp_matrix.csv")
print("Reading Methylation...")
with open("../../data/raw_data/methylation_mean.dat", 'rb') as f:
    methy = pickle.load(f)
    f.close()

meta_data = pd.DataFrame(columns=['id', 'type', 'image_path', 'flatten_path','tp53','met','age', 'gender', 'wGII'])
for index, row in tqdm(clinical.iterrows()):
    id = row['bcr_patient_barcode']
    type = row['type']
    if type == None:
        type = -1
    met = row['metastatic_one_two_three']
    if math.isnan(met):
        met = -1
    age = row['age_at_initial_pathologic_diagnosis']
    if math.isnan(age):
        age = -1
    gender = row['gender']
    if gender == None:
        gender = -1
    wGII = row['wGII']
    if math.isnan(wGII):
        wGII = -1
    tmp_mut = muts[muts["sampleID"] == id]
    tp53 = -1
    if "TP53" in tmp_mut['Hugo_Symbol'].values:
        tp53 = 1
    else:
        tp53 = 0
    if args.tp53:
        print("Filtering tp53 from data")
        tmp_mut = tmp_mut[tmp_mut['Hugo_Symbol'] != "TP53"]

    if (args.tp53 and tp53 in [0, 1]) or (not args.tp53 and met in [0, 1]):
        print("\tMaking image")
        image = make_image_chr(id, met, all_genes)
        print("\tMapping losses to genes")
        image = find_losses(id, image, all_genes, ascat_loss)
        print("\tMapping gains to genes")
        image = find_gains(id, image, all_genes, ascat_gain)
        print("\tMapping mutations to genes")
        image = find_mutations(id, image, muts)
        print("\tMapping expression to genes")
        image = find_gene_expression(id, image, gene_exp,
                                     np.min(np.array(gene_exp.select_dtypes(include=np.number))),
                                     np.max(np.array(gene_exp.select_dtypes(include=np.number))))
        print("\tMapping methylation to genes")
        image = find_methylation(id, image, methy)
        #print("\tStoring intermediate results in .dat binary file...")
        # with open("../../data/{}/dictionary_images/{}.dat".format(folder, id), 'wb') as f:
        #     pickle.dump(image, f)
        #     f.close()
        image.make_image_matrces_by_chr()
        n_dim_image = image.make_n_dim_chr_image()
        print("\tStoring n dim image in .dat file")
        with open("../../data/{}/n_dim_images/{}.dat".format(folder, id), 'wb') as f:
            pickle.dump(n_dim_image, f)
            f.close()

        meta_data = meta_data.append({'id': str(id),
                                      'type': str(type),
                                      'image_path': str("n_dim_images/{}.dat".format(id)),
                                      'flatten_path': str("flatten_vectors/{}.dat".format(id)),
                                      'tp53': int(tp53),
                                      'met': int(met),
                                      'age': int(age),
                                      'gender': str(gender),
                                      'wGII': float(wGII)
                                      },
                                     ignore_index=True)
meta_data.to_csv("../../data/{}/meta_data.csv".format(folder))
print("Done in --- %s seconds ---" % (time.time() - start_time))
