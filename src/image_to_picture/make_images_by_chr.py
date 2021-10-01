import pickle
import time

import numpy as np
import pandas as pd
from utils import make_image_chr, find_mutations, find_losses, find_gains, \
    find_gene_expression

from TCGA_GenomeImage.src.image_to_picture.utils import find_methylation


folder = 'TP53_data/Random22x3760Image/'

start_time = time.time()
print("Reading clinical...")
clinical = pd.read_csv("../../data/raw_data/clinical.csv")
print("Reading ascat...")
ascat = pd.read_csv("../../data/raw_data/ascat.csv")
ascat_loss = ascat.loc[ascat['loss'] == True]
ascat_gain = ascat.loc[ascat['gain'] == True]
print("Reading all gene definition...")
all_genes = pd.read_csv("../../data/raw_data/all_genes_ordered_by_chr.csv")
all_genes = all_genes.sample(frac=1).reset_index(drop=True)
all_genes = all_genes[all_genes['name2'] != "TP53"]
print("Reading Muts...")
muts = pd.read_csv("../../data/raw_data/muts.csv")
print("Reading gene exp...")
gene_exp = pd.read_csv("../../data/raw_data/gene_exp_matrix.csv")
print("Reading Methylation...")
with open("../../data/raw_data/methylation_mean.dat", 'rb') as f:
    methy = pickle.load(f)
    f.close()

meta_data = pd.DataFrame(columns=['id', 'type', 'image_path', 'flatten_path', 'hilbert_path', 'tp53','met'])
for index, row in clinical.iterrows():
    id = row['bcr_patient_barcode']
    type = row['type']
    met = row['metastatic_one_two_three']
    tmp_mut = muts[muts["sampleID"] == id]
    tp53 = -1
    if "TP53" in tmp_mut['Hugo_Symbol'].values:
        tp53 = 1
    else:
        tp53 = 0
    print(id, "->", tp53)
    print(tmp_mut.shape)
    print("Filtering TP53 muts from data")
    tmp_mut = tmp_mut[tmp_mut['Hugo_Symbol'] != "TP53"]
    print(tmp_mut.shape)
    print(id, "->", met)
    if tp53 in [0, 1]:
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
        print("\tStoring intermediate results in .dat binary file...")
        with open("../../data/{}/dictionary_images/{}.dat".format(folder, id), 'wb') as f:
            pickle.dump(image, f)
            f.close()
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
                                      'hilbert_path': str("hilbert_transforms/no_exist.dat"),
                                      'tp53': int(tp53),
                                      'met': int(met)
                                      },
                                     ignore_index=True)

meta_data.to_csv("../../data/{}/meta_data.csv".format(folder))
print("Done in --- %s seconds ---" % (time.time() - start_time))
