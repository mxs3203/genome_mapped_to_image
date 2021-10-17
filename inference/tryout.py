import pickle
import pandas as pd
import pyreadr as pr


cin = pr.read_r("../data/Metastatic_data/193x193Image/All_TCGA_CIN_measures.rds")
cin = cin[None]
meta = pd.read_csv("../data/meta_data.csv")
clin = pd.read_csv("../data/TCGA_survival_data_clean.txt", sep='\t')

print(clin['age_at_initial_pathologic_diagnosis'])

#meta = pd.merge(left =meta, right=clin[['bcr_patient_barcode','age_at_initial_pathologic_diagnosis']],
#         left_on='id', right_on='bcr_patient_barcode')
meta = pd.merge(left =meta, right=cin[['sample_id','wGII']],
         left_on='id', right_on='sample_id')


print(meta)

meta.to_csv("../data/meta_data_wgii.csv")