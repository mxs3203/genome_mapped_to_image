import argparse

import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

from src.image_to_picture.utils import make_image, find_losses, find_gains, find_mutations, find_gene_expression, \
    find_methylation

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output',type=str, default='TCGA_Square_ImgsGainLoss_harsh/Metastatic_data/SquareImg')
parser.add_argument('--tp53', type=int, default='0')
parser.add_argument('--shuffle', type=int, default='0')

args = parser.parse_args()
args.tp53 = bool(int(args.tp53))
args.shuffle = bool(int(args.shuffle))
folder = args.output #'TP53_data/ShuffleImg'
print(args)
start_time = time.time()
print("Reading clinical...")
clinical = pd.read_csv("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/corrected_metastatic_based_on_stages.csv")
print("Reading ascat...")
ascat = pd.read_csv("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/ascat.csv")
ascat_loss = ascat.loc[ascat['loss'] == True]
ascat_gain = ascat.loc[ascat['gain'] == True]
print("Reading all gene definition...")
all_genes = pd.read_csv("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/all_genes_ordered_by_chr_no_sex_chr.csv")

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

for index, row in clinical.iterrows():

    id = row['id']
    print(index, "/", clinical.shape[0], "  ", id)
    met = row['metastatic_one_two_three']

    tmp_mut = muts[muts["sampleID"] == id]

    if args.tp53:
        print("Filtering tp53 from data")
        tmp_mut = tmp_mut[tmp_mut['Hugo_Symbol'] != "TP53"]

    print("\tMaking image")
    image = make_image(id, met, all_genes)
    print("\tMapping losses to genes")
    #image = find_losses(id, image, all_genes, ascat_loss)
    print("\tMapping gains to genes")
    #image = find_gains(id, image, all_genes, ascat_gain)
    print("\tMapping mutations to genes")
    image = find_mutations(id, image, tmp_mut)
    print("\tMapping expression to genes")
    image = find_gene_expression(id, image, gene_exp,
                                 np.min(np.array(gene_exp.select_dtypes(include=np.number))),
                                 np.max(np.array(gene_exp.select_dtypes(include=np.number))))
    print("\tMapping methylation to genes")
    image = find_methylation(id, image, methy)
    image.make_image_matrces()
    # five_dim_image = image.make_5_dim_image()
    # feature_vector = image.vector_of_all_features()
    # if np.all((feature_vector == 0)):
    #     print("All zeros in 5d, not saving...")
    # else:
    #     print("\tStoring n dim images in .dat file")
    #     with open("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/{}/5_dim_images/{}.dat".format(folder, id),
    #               'wb') as f:
    #         pickle.dump(five_dim_image, f)
    #         f.close()
    #     with open("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/{}/flatten_vectors_5d/{}.dat".format(folder, id),
    #               'wb') as f:
    #         pickle.dump(feature_vector, f)
    #         f.close()

    three_dim_image = image._3channel_square()
    three_dim_image_feature_vector = image._3channel_flat()
    if np.all((three_dim_image_feature_vector == 0)):
        print("All zeros in 3d vector, not saving...")
    else:
        with open("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/{}/3_dim_images/{}.dat".format(folder, id), 'wb') as f:
            pickle.dump(three_dim_image, f)
            f.close()
        with open("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/{}/flatten_vectors_3d/{}.dat".format(folder, id), 'wb') as f:
            pickle.dump(three_dim_image_feature_vector, f)
            f.close()




print("Done in --- %s minutes ---" % ((time.time() - start_time) / 60))
