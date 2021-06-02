import pickle

with open("../data/n_dim_images/TCGA-2E-A9G8.dat", 'rb') as f:
    image = pickle.load(f)
    f.close()


print(image)