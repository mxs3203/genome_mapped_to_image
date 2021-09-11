import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class TCGAImageLoader(Dataset):

    def __init__(self, csv_file,response_var='tp53',folder_name='TP53_data',
                 image_type='193x193Image', response_column_index=-1,
                 predictor_column_index = -1,
                 filter_by_type=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.annotation = pd.read_csv(csv_file, sep=",")
        if filter_by_type is not None:
            self.annotation = self.annotation[self.annotation['type'] == filter_by_type ]
            self.annotation = self.annotation[self.annotation[response_var] == 1]

        self.transform = transform
        self.folder_name = folder_name
        self.image_type = image_type
        self.response_column_index = response_column_index
        self.predictor_column_index = predictor_column_index

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open("../data/{}/{}/{}".format(self.folder_name, self.image_type, self.annotation.iloc[idx, self.predictor_column_index]), 'rb') as f:
            image = pickle.load(f)
            f.close()
        met_1_2_3 = np.array(self.annotation.iloc[idx, self.response_column_index], dtype="long")
        if self.transform:
            image = self.transform(image)

        return image,self.annotation.iloc[idx, 2], self.annotation.iloc[idx, 1], met_1_2_3