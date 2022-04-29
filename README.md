# Genome mapped to Image (GMI) Project
### Mapping entire genome information to the image and learning the features of the genome as image. 

### Step 1: Organize raw data in data/raw_data folder \
* Check example input 
### Step 2: Map genome data to images
* First we enter the directory with the scripts for making images \
* We run the script for every image transofrmation we want: \
* For example: \
    * &ensp; TP53 images need to remove the TP53 gene from the list while making images therefore we put --tp53 1 \
    * &ensp; If we want to created shuffled image with random position of genes we use --shuffle 1, otherwise 0 \
* You can run this code in parallel if you have enough memory and CPU power
```
cd src/image_to_picture
python3 make_images.py --output data/Metastatic_data --tp53 0 --shuffle 0
python3 make_images.py --output data/Metastatic_data --tp53 0 --shuffle 1
python3 make_images.py --output data/TP53_data --tp53 1 --shuffle 1
python3 make_images.py --output data/TP53_data --tp53 1 --shuffle 0
```
* If you want to build Chromosome organized images use the following code
```
python3 make_images_by_chr.py --output data/TP53_data --tp53 1 --shuffle 0
python3 make_images_by_chr.py --output data/Metastatic_data --tp53 0 --shuffle 0
```
* make_images and make_image_by_chr will also make a manifest file which you need for training.
Those files link image path with sample ID and response variables

### Step 3: Prepare for Training  
* Take AutoEncoder.AE.AE class if you are working with Chromosome Organized Images
* Take AutoEncoder.AE.AE_Square if you are working with rectangular Images
* If you are predicting numerical response use train_num_response.py and for categorical use train.py
* Write config file which tells the training which hyperparameters to use and which folders to use

### Step 4: Training the model
* In this step we train the model for predicting metastatic disease by running the following code:
```
python3 train.py config/metastatic
```
* In this step we train the model for predicting wGII (numeric value) by running the following code:
```
python3 train_num_response.py config/wGII
```
### Step 5: Inspect the model performance 
* If you are happy with model performance continue with the next step
* If you do not "trust" the model because it does not predict outcome variable with high certainty, try tweaking the model parameters or hyperparameters.

### Step 6: Integrated Gradients (IG)
 * The training script outputs model with the best loss in .pb format
 * This file is needed for IG step
 * We run the following script to get the gradients of a model
```
cd ../../inference
python3 analyze_network.py
```
* This script will produce csv files representing attribution for every input

### Step 7: Encode genome data
* We can also use trained model to encode entire genome image to a vector of length 128
* This can be used in further analysis
* We run the following script to exstract encoded genomes
```
python3 encode_genome_image.py
``` 
### Step 8: Downstream Analysis
* Downstream analysis of attribution genes
* Downstream analysis of encoded genomes



