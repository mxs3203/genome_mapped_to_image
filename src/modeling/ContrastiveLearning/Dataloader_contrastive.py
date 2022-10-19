import glob
import pickle
import random
from collections import Counter
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from torchvision.transforms import transforms


class TCGAImageLoaderContrastive(Dataset):

    def __init__(self, csv_file,  folder, image_type, predictor_column, response_column, filter_by_type=None, transform=None ):

        self.annotation = pd.read_csv(csv_file, sep=",")
        if filter_by_type is not None:
            self.annotation = self.annotation[self.annotation.type.isin(filter_by_type)]
        self.number_of_c_types = len(self.annotation['type'].unique())
        ord_enc = OrdinalEncoder()
        scaler = MinMaxScaler()
        self.annotation["type_coded"] = ord_enc.fit_transform(self.annotation[["type"]])
        self.annotation["gender_coded"] = ord_enc.fit_transform(self.annotation[["gender"]])
        self.annotation["type_coded_random"] = np.random.randint(0,self.number_of_c_types, size=np.shape(self.annotation)[0])
        self.annotation["age_scaled"] = scaler.fit_transform(self.annotation[["age"]])
        #self.annotation["stage_coded"] = ord_enc.fit_transform(self.annotation[["final_stage"]])
        self.f_names = pd.unique(self.annotation['type'])
        self.transform = transform
        self.folder = folder
        self.image_type = image_type
        self.predictor_column = predictor_column
        self.response_column = response_column
        self.remove_rows_where_there_is_no_file()

    def compute_class_weight(self, dataset):
        y = []
        for x,y_label,i in dataset:
            y.append(y_label.item())
        count = Counter(y)
        class_count = np.array([count[0], count[1]])
        weight = 1. / class_count
        samples_weight = np.array([weight[t] for t in y])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def remove_rows_where_there_is_no_file(self):
        print("Finding all files from metadata... number of files: ",np.shape(self.annotation)[0])
        files = glob.glob("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/{}/{}/5_dim_images/*.dat".format(self.folder, self.image_type))
        ids = [f.split("/")[10] for f in files]
        ids = [f.split(".")[0] for f in ids]
        self.annotation = self.annotation[self.annotation['id'].isin(ids)]
        print("Number of Files after removing the missing files: ",np.shape(self.annotation)[0])

    def __len__(self):
        return len(self.annotation)

    def add_noise_to_layer(self, x, layer, random_changes=2000):
        x_layer = x[layer, : , :]
        mask = np.random.randint(0, random_changes, size=x_layer.shape).astype(np.bool)
        r = np.random.rand(*x_layer.shape) * np.max(x_layer)
        x_layer[mask] = r[mask]
        x[layer, :, :] = x_layer
        return x

    def augment(self,x, prob=50, layers=5):
        for i in range(layers):
            if random.randrange(0, 100) < prob:
                x = self.add_noise_to_layer(x, i)
        if random.randrange(0, 100) < prob:
            x = self.gauss_noise(x)
        return x

    def gauss_noise(self,x):
        return gaussian_filter(x, sigma=random.uniform(0.1, 2))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # flatten_vectors_5d, 5_dim_images
        with open("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/{}/{}/5_dim_images/{}.dat".format(self.folder, self.image_type, self.annotation.iloc[idx, self.predictor_column]), 'rb') as f:
            x = pickle.load(f)
            f.close()
        y = np.array(self.annotation.iloc[idx, self.response_column], dtype="long")

        x1 = self.augment(x)
        x2 = self.augment(x)

        return x1,x2,y