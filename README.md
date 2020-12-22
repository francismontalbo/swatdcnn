# Automated Diagnosis of Diverse Coffee Leaf Images through a Stage-Wise Aggregated Triple Deep Convolutional Neural Network
Stage Wise Aggregated Triple Deep Convolutional Neural Network (SWAT-DCNN)
Author: [Francis Jesmar P. Montalbo](https://francismontalbo.github.io) 

## Graphical Abstract

<img src="/graphical_abstract_diverse_coffee.jpg" alt="Stage Wise Aggregated Triple Deep Convolutional Neural Network for Coffee Leaf Diagnosis" width="1000">


## DATASETS

The dataset used for this work came from the following works:
**Please consider citing their work when using it** 

**RoCoLe** https://data.mendeley.com/datasets/c5yvn32dzg/2
Parraga-Alava, Jorge; Cusme, Kevin; Loor, Angélica; Santander, Esneider (2019), 
“RoCoLe: A robusta coffee leaf images dataset ”, 
Mendeley Data, V2, doi: 10.17632/c5yvn32dzg.2
http://dx.doi.org/10.17632/c5yvn32dzg.2

Inclusion: 
- [x] Healthy
- [x] Coffee Leaf Rust (CLR)
- [x] Red Spider Mites (RSM) 

**BrACoL** https://data.mendeley.com/datasets/yy2k5y8mxg
Krohling, Renato; esgario, José; Ventura, Jose A. (2019),
“BRACOL - A Brazilian Arabica Coffee Leaf images dataset to identification and quantification of coffee diseases and pests”, 
Mendeley Data, V1, doi: 10.17632/yy2k5y8mxg.1
http://dx.doi.org/10.17632/yy2k5y8mxg.1

https://www.sciencedirect.com/science/article/abs/pii/S0168169919313225
Esgario, J. G., Krohling, R. A., & Ventura, J. A. (2020). 
Deep learning for classification and severity estimation of coffee leaf biotic stress. 
Computers and Electronics in Agriculture, 169, 105162.

Inclusion: 
- [x] Healthy
- [x] CLR
- [x] Cercospora Leaf Spots (CLS)
- [x] Phoma Leaf Spots (PLS)
- [x] Coffee Leaf Miner (CLM)

**LiCoLe** https://ijain.org/index.php/IJAIN/article/view/495/0
MONTALBO, Francis Jesmar Perez; HERNANDEZ, Alexander Arsenio. 
Classifying Barako coffee leaf diseases using deep convolutional models. 
International Journal of Advances in Intelligent Informatics, 
[S.l.], v. 6, n. 2, p. 197-209, july 2020. ISSN 2548-3161. 
Available at: <https://ijain.org/index.php/IJAIN/article/view/495%7Cto_array%3A0>. 
doi:https://doi.org/10.26555/ijain.v6i2.495.

Inclusion: 
- [x] Healthy
- [x] CLR
- [x] Sooty Molds (SM)

#### For the readily prepared dataset used in this work refer to this link
** Google Drive Link **

** Note ** The following dataset credits still goes to their appropriate owners and collectors

## Environment Setup:

Make sure to create a new virtual environment preferably in Anaconda. Use Python 3.5.

https://www.anaconda.com/

https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

Activate and access the folder `swatdcnn/` with the included `requirements.txt` file.

**This uses a tensorflow GPU**

Afterwards, simply enter the command in the conda CLI `pip install -r requirements.txt`

Dependencies include: 
- efficientnet==1.1.1
- colorama==0.4.3
- jupyter==1.0.0
- keras==2.2.5
- numpy==1.16.2
- opencv-python==4.4.0.42
- pandas==0.25.3
- Pillow==7.2.0
- scikit-learn
- scikit-image
- scikit-plot
- scipy
- tensorflow-gpu==1.14.0

Once installed, you may either train the models individually with the `.ipynb` notebooks found in `swatdcnn/models/` or make use of the pre-trained weights on the give link.

**Pre-trained weights required by the SWAT-DCNN

**T-DCNN Weights Link

Make sure to save the links under the `/models/` folder.

## How to use:

**Training with the pre-trained weights (RECOMMENDED)** 

1. Run the system by going to the swatdcnn folder with your created virtual environment activated and enter the command `python swatdcnn.py`

2. Follow through the given instructions and make sure to use the test sample from the provided `/test/` folder

**Training from scratch**

1. Activate your created virtual environment and enter the main `swatdcnn/` folder.

2. Save the dataset folder downloaded from LINK inside the `swatdcnn/` as `swatdcnn/dataset/`

3. Open the `.ipynb` files from the `swatdcnn/models` folder and run the following in your preferred order. The `swatdcnn/models/tdcnn/` is saved for later.

4. Once all models from stage-1 to 3 are trained. You may now open the `swatdcnn/models/tdcnn/` folder to build the T-DCNN models.

5. After all T-DCNN models are built, you may now run the `swatdcnn.py` from the main `swatdcnn/` folder.

6. Follow through the given instructions and make sure to use the test sample from the provided `/test/` folder



**In case of any problems, don't hesitate to contact me. I'll be happy to help.**

## Citation

This GitHub repository serves as a support for the submitted publication article in **_UNDER REVIEW_**

[Author Profile](https://scholar.google.com/citations?user=PV8dJDkAAAAJ&hl=en&oi=ao)




