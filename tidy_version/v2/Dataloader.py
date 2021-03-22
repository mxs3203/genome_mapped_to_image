import os

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class TCGAImageLoader(Dataset):

    def __init__(self, csv_file, root_dir,filter_by_type=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.annotation = pd.read_csv(csv_file, sep=";")
        if filter_by_type is not None:
            self.annotation = self.annotation[self.annotation['type'] == filter_by_type ]
            self.annotation = self.annotation[self.annotation['metastatic_one_two_three'] == 1]

        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cin_loss = os.path.join(self.root_dir, self.annotation.iloc[idx, 3])
        cin_loss = pd.read_csv(cin_loss, sep=",", dtype="float32",header=None, names = None)
        cin_loss = np.asarray(cin_loss, dtype="float32")
        cin_gain = os.path.join(self.root_dir, self.annotation.iloc[idx, 4])
        cin_gain = pd.read_csv(cin_gain, sep=",", dtype="float32",header=None, names = None)
        cin_gain = np.asarray(cin_gain, dtype="float32")
        mut_file = os.path.join(self.root_dir, self.annotation.iloc[idx, 5])
        mut = pd.read_csv(mut_file, sep=",", dtype="float32",header=None, names = None)
        mut = np.asarray(mut, dtype="float32")
        image = np.dstack((cin_gain, cin_loss,  mut))

        dss = np.array(self.annotation.iloc[idx, 6], dtype="long")
        type = self.annotation.iloc[idx, 7]
        met_1_2 = np.array(self.annotation.iloc[idx, 8], dtype="long")
        met_1_2_3 = np.array(self.annotation.iloc[idx, 9], dtype="long")
        GD = np.array(self.annotation.iloc[idx, 10], dtype="long")
        if self.transform:
            image = self.transform(image)


        return image,dss,type, self.annotation.iloc[idx, 1], met_1_2, met_1_2_3, GD