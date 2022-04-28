import pickle
import random

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler



class TCGAImageLoader(Dataset):

    def __init__(self, csv_file,  folder, image_type, predictor_column, response_column, filter_by_type=None, transform=None ):

        self.annotation = pd.read_csv(csv_file, sep=",")
        if filter_by_type is not None:
            self.annotation = self.annotation[self.annotation.type.isin(filter_by_type)]
        self.number_of_c_types = len(self.annotation['type'].unique())
        ord_enc = OrdinalEncoder()
        scaler = MinMaxScaler()
        self.annotation["type_coded"] = ord_enc.fit_transform(self.annotation[["type"]])
        self.annotation["gender_coded"] = ord_enc.fit_transform(self.annotation[["gender"]])
        self.annotation["type_coded_random"] = np.random.randint(0,5, size=np.shape(self.annotation)[0])
        self.annotation["age_scaled"] = scaler.fit_transform(self.annotation[["age"]])
        self.f_names = pd.unique(self.annotation['type'])
        self.transform = transform
        self.folder = folder
        self.image_type = image_type
        self.predictor_column = predictor_column
        self.response_column = response_column

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/{}/{}/{}".format(self.folder, self.image_type, self.annotation.iloc[idx, self.predictor_column]), 'rb') as f:
            x = pickle.load(f)
            f.close()
        y = np.array(self.annotation.iloc[idx, self.response_column], dtype="long")
        if self.transform:
            x = self.transform(x)

        return x, y, self.annotation.iloc[idx,0]