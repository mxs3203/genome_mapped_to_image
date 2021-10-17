import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing


class TCGAImageLoader(Dataset):

    def __init__(self, csv_file,  folder, image_type, predictor_column, response_column, filter_by_type=None, transform=None ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        le = preprocessing.LabelEncoder()
        self.annotation = pd.read_csv(csv_file, sep=",")
        if filter_by_type is not None:
            self.annotation = self.annotation[self.annotation.type.isin(filter_by_type)]

        self.annotation['encoded_type'] = le.fit_transform(self.annotation['type'])
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

        with open("../../data/{}/{}/{}".format(self.folder,self.image_type,self.annotation.iloc[idx, self.predictor_column]), 'rb') as f:
            x = pickle.load(f)
            f.close()
        y = np.array(self.annotation.iloc[idx, self.response_column], dtype="float")
        if self.transform:
            x = self.transform(x)

        return x,self.annotation.iloc[idx, 2], self.annotation.iloc[idx, 1], y