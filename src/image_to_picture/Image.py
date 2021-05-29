import numpy as np

class Image:
    def __init__(self, id,dict_of_cells,met):
        self.id = id
        self.dict_of_cells = dict_of_cells
        self.met = met

        self.gain_matrix = None
        self.loss_matrix = None
        self.methy_matrix = None
        self.mut_matrix = None
        self.exp_matrix = None

    def make_image_matrces(self):
        loss = np.zeros((193, 193))
        gain = np.zeros((193, 193))
        mut = np.zeros((193, 193))
        methy = np.zeros((193, 193))
        exp = np.zeros((193, 193))

        for gene in self.dict_of_cells:
            if self.dict_of_cells[gene].loss_val is not None:
                loss[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].loss_val
            if self.dict_of_cells[gene].gain_val is not None:
                gain[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].gain_val
            if self.dict_of_cells[gene].mut_val is not None:
                mut[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].mut_val
            if self.dict_of_cells[gene].exp_val is not None:
                exp[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].exp_val
            if self.dict_of_cells[gene].methy_val is not None:
                methy[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].methy_val
        self.loss_matrix = np.asarray(loss, dtype="float32")
        self.exp_matrix = np.asarray(exp, dtype="float32")
        self.gain_matrix = np.asarray(gain, dtype="float32")
        self.mut_matrix = np.asarray(mut, dtype="float32")
        self.methy_matrix = np.asarray(methy, dtype="float32")

    def make_n_dim_image(self):
        image = np.stack((self.gain_matrix, self.loss_matrix, self.mut_matrix, self.exp_matrix))
        #image = image.reshape((4,193,193))
        return image

