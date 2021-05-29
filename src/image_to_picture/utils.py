from TCGA_GenomeImage.src.image_to_picture.Image import Image
from TCGA_GenomeImage.src.image_to_picture.ImageCell import ImageCell
import pandas as pd
import numpy as np

def normalize_data(data, min, max):
    return (data - min) / (max - min)

def make_image(id, met, all_genes):
    cnt = 0
    dict = {}
    for i in range(193):
        for j in range(193):
            if cnt < all_genes.shape[0]:
                img = ImageCell(all_genes['name2'].iloc[cnt],loss_val=None,gain_val=None,mut_val=None,exp_val=None,methy_val=None, chr=all_genes['chr'].iloc[cnt])
                img.i = i
                img.j = j
                dict[all_genes['name2'].iloc[cnt]] = img
            else:
                img = ImageCell(None, None, None, None, None, None, None)
                img.i = i
                img.j = j
            cnt += 1
        cnt += 1
    return Image(id=id, met=met, dict_of_cells=dict)

def find_mutations(id,image, muts):
    if id in np.array(muts['sampleID']):
        muts_tmp = muts.loc[muts['sampleID'] == id]
        for i, row in muts_tmp.iterrows():
            if row['Hugo_Symbol'] in image.dict_of_cells:
                image.dict_of_cells[row['Hugo_Symbol']].mut_val = row['PolyPhen_num']
        #print("\tFound ", muts_tmp.shape[0], "genes affected by mutation")
    return image

def find_gene_expression(id,image, gene_exp, min,max):
    if id in np.array(gene_exp.columns):
        genes = gene_exp['gene']
        exp = gene_exp[id]
        for i in range(len(genes)):
            if genes[i] in image.dict_of_cells:
                #print(exp[i], " -> ", normalize_data(exp[i], min,max))
                image.dict_of_cells[genes[i]].exp_val = normalize_data(exp[i], min,max)
    return image


def find_losses(id, image, all_genes, ascat_loss):
    if id in np.array(ascat_loss['ID']):
        ascat_loss_tmp = ascat_loss.loc[ascat_loss['ID'] == id]
        for i, row in ascat_loss_tmp.iterrows():
            seg_end = row['End']
            seg_start = row['Start']
            # find all affected genes
            genes_affected_full = all_genes['name2'][((all_genes['start'] >= seg_start) & (all_genes['end'] <= seg_end))]
            genes_affected_partial1 = all_genes['name2'][all_genes['start'].between(seg_start, seg_end, inclusive=True)]
            genes_affected_partial2 = all_genes['name2'][all_genes['end'].between(seg_start, seg_end, inclusive=True)]

            genes_affected = pd.concat([genes_affected_full,genes_affected_partial1,genes_affected_partial2])
            # print("\tFound ", len(genes_affected_full), "genes affected by full loss")
            # print("\tFound ", len(genes_affected_partial1), "genes affected by partial loss(start)")
            # print("\tFound ", len(genes_affected_partial2), "genes affected by partial loss(end)")
            for g in genes_affected:
                if g in image.dict_of_cells:
                    #print(row['log_r'], " -> ", normalize_data(row['log_r'],  ascat_loss['log_r'].max(), ascat_loss['log_r'].min()))
                    image.dict_of_cells[g].loss_val = normalize_data(row['log_r'], ascat_loss['log_r'].max(),ascat_loss['log_r'].min())

    return image


def find_gains(id, image, all_genes, ascat_gains):
    if id in np.array(ascat_gains['ID']):
        ascat_loss_tmp = ascat_gains.loc[ascat_gains['ID'] == id]
        for i, row in ascat_loss_tmp.iterrows():
            seg_end = row['End']
            seg_start = row['Start']
            # find all affected genes
            genes_affected = all_genes['name2'][((all_genes['start'] >= seg_start) & (all_genes['end'] <= seg_end))]
            #print("\tFound ", len(genes_affected), "genes affected by gain")
            for g in genes_affected:
                if g in image.dict_of_cells:
                    #print(row['log_r'], " -> ", normalize_data(row['log_r'], ascat_gains['log_r'].min(), ascat_gains['log_r'].max()))
                    image.dict_of_cells[g].gain_val = normalize_data(row['log_r'], ascat_gains['log_r'].min(), ascat_gains['log_r'].max())

    return image


def find_methylation(id,image, all_genes, methy):
    if id in np.array(methy.columns):
        for i,row in all_genes.iterrows():
            gene = row['name2']
            met_g1 = np.array(methy['gene1'])
            if gene in image.dict_of_cells and gene in met_g1:
                vals = methy[id][methy['gene1'] == row['name2']]
                image.dict_of_cells[gene].methy_val = vals.mean()
    return image

# TODO: save images of each layer as example

def makeImages(x):
    img_cin_g = x[0, 0, :, :]
    img_cin_g = img_cin_g.astype('uint8')
    img_cin_l = x[0, 1, :, :]
    img_cin_l = img_cin_l.astype('uint8')
    img_mut = x[0, 2, :, :]
    img_mut = img_mut.astype('uint8')
    img_mut = Image.fromarray(img_mut, 'P')
    img_cin_g = Image.fromarray(img_cin_g, 'P')
    img_cin_l = Image.fromarray(img_cin_l, 'P')
    total_img = np.dstack((img_cin_g, img_cin_l, img_mut))
    total_img = Image.fromarray(total_img, 'RGB')
    img_cin_g.save("cin_gain.png")
    img_cin_l.save("cin_loss.png")
    img_mut.save("mut.png")
    total_img.save("total.png")
    return img_cin_l, img_cin_g, img_mut