import pickle


with open("../../data/n_dim_images/TCGA-2F-A9KO.dat", 'rb') as f:
    new_class = pickle.load(f)

print(new_class)

