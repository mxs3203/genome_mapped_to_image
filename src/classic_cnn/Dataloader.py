import pickle
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing


class TCGAImageLoader(Dataset):

    def __init__(self, csv_file,  folder, image_type, predictor_column, response_column, filter_by_type=None, transform=None ):

        self.annotation = pd.read_csv(csv_file, sep=",")
        if filter_by_type is not None:
            self.annotation = self.annotation[self.annotation.type.isin(filter_by_type)]

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

        with open("../../data/{}/{}/{}".format(self.folder,self.image_type, self.annotation.iloc[idx, self.predictor_column]), 'rb') as f:
            x = pickle.load(f)
            f.close()
        y = np.array(self.annotation.iloc[idx, self.response_column], dtype="long")
        if self.transform:
            x = self.transform(x)

        return x, y