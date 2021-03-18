import os

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class TCGAImageLoader(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        type_list = ['BLCA', 'PRAD', 'OV']
        self.annotation = pd.read_csv(csv_file, sep=";")
        self.annotation = self.annotation.loc[self.annotation ['type'].isin(type_list)]
        self.root_dir = root_dir
        self.transform = transform

    def normalizeImg(self, cin, mut):
        cin_gain = cin.copy()
        cin_loss = cin.copy()
        cin_gain[cin_gain < 0] = 0
        cin_loss[cin_loss > 0] = 0
        cin_loss = np.abs(cin_loss)
        mut[mut == 1] = 254
        mut = mut + 1
        cin_loss = ((cin_loss - cin_loss.min()) * (1 / (cin_loss.max()+0.01 - cin_loss.min()) * 255)).astype('uint8')
        cin_gain = ((cin_gain - cin_gain.min()) * (1 / (cin_gain.max()+0.01 - cin_gain.min()) * 255)).astype('uint8')

        return cin_loss, cin_gain, mut
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cin_file = os.path.join(self.root_dir, self.annotation.iloc[idx, 2])
        cin = pd.read_csv(cin_file, sep=",", dtype="float32")
        cin = np.asarray(cin, dtype="float32")
        mut_file = os.path.join(self.root_dir, self.annotation.iloc[idx, 3])
        mut = pd.read_csv(mut_file, sep=",", dtype="float32")
        mut = np.asarray(mut, dtype="float32")
        cin_loss,cin_gain, mut = self.normalizeImg(cin, mut)
        image = np.dstack((cin_loss,cin_gain, mut))
        image = np.reshape(image, (198, 197, 3))
        dss = np.array(self.annotation.iloc[idx, 4], dtype="long")
        type = self.annotation.iloc[idx, 5]
        if self.transform:
            image = self.transform(image)


        return image,dss,type, self.annotation.iloc[idx, 1]