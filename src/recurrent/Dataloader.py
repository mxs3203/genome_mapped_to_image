import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class TCGAImageLoader(Dataset):

    def __init__(self, csv_file, filter_by_type=None, transform=None):
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
            self.annotation = self.annotation[self.annotation['met'] == 1]

        self.transform = transform


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open("../../data/images_by_chr/{}".format(self.annotation.iloc[idx, 3]), 'rb') as f:
            image = pickle.load(f)
            f.close()
        met_1_2_3 = np.array(self.annotation.iloc[idx, 4], dtype="long")
        if self.transform:
            image = self.transform(image)

        return image,self.annotation.iloc[idx, 2], self.annotation.iloc[idx, 1], met_1_2_3