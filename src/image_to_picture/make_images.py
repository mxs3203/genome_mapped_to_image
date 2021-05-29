import pandas as pd
import numpy as np
import pickle
import time
from TCGA_GenomeImage.src.image_to_picture.utils import make_image, find_gains, find_losses, find_mutations, \
    find_gene_expression, find_methylation
start_time = time.time()
print("Reading clinical...")
clinical = pd.read_csv("../../data/raw_data/clinical.csv")
print("Reading ascat...")
ascat = pd.read_csv("../../data/raw_data/ascat.csv")
ascat_loss = ascat.loc[ascat['loss'] == True]
ascat_gain = ascat.loc[ascat['gain'] == True]
print("Reading all gene definition...")
all_genes = pd.read_csv("../../data/raw_data/all_genes_ordered_by_chr.csv")
print("Reading Muts...")
muts = pd.read_csv("../../data/raw_data/muts.csv")
print("Reading gene exp...")
gene_exp = pd.read_csv("../../data/raw_data/gene_exp_matrix.csv")
#print("Reading Methylation...")
#methy = pd.read_csv("../../data/raw_data/methylation.csv")

meta_data = pd.DataFrame(columns=['id', 'type', 'image_path', 'met'])
for index, row in clinical.iterrows():
    id = row['bcr_patient_barcode']
    type = row['type']
    met = row['metastatic_one_two_three']
    print(id, "->",met)
    if met in [0,1] :
        print("\tMaking image")
        image = make_image(id, met, all_genes)
        print("\tMapping losses to genes")
        image = find_losses(id, image, all_genes, ascat_loss)
        print("\tMapping gains to genes")
        image = find_gains(id, image, all_genes, ascat_gain)
        print("\tMapping mutations to genes")
        image = find_mutations(id, image, muts)
        print("\tMapping expression to genes")
        image = find_gene_expression(id,image, gene_exp,
                                     np.min(np.array(gene_exp.select_dtypes(include=np.number))),
                                     np.max(np.array(gene_exp.select_dtypes(include=np.number))))
        #print("\tMapping methylation to genes")
        #image = find_methylation(id, image,all_genes, methy)
        # print("\tStoring intermediate results in .dat binary file...")
        # with open("../../data/dictionary_images/{}.dat".format(id), 'wb') as f:
        #     pickle.dump(image, f)
        #     f.close()
        image.make_image_matrces()
        n_dim_image = image.make_n_dim_image()
        print("\tStoring n dim image in .dat file")
        with open("../../data/n_dim_images/{}.dat".format(id), 'wb') as f:
            pickle.dump(n_dim_image, f)
            f.close()
        meta_data = meta_data.append({'id': str(id),
                                      'type': str(type),
                                      'image_path': str("n_dim_images/{}.dat".format(id)),
                                      'met': int(met)
                                      },
                                     ignore_index=True)

meta_data.to_csv("../../data/meta_data.csv")
print("Done in --- %s seconds ---" % (time.time() - start_time))





