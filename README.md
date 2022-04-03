# Automated Diagnosis of Diverse Coffee Leaf Images through a Stage-Wise Aggregated Triple Deep Convolutional Neural Network :leaves: :coffee:
**Stage Wise Aggregated Triple Deep Convolutional Neural Network (SWAT-DCNN)**
<p>Author: Francis Jesmar P. Montalbo</p>
<p>Affiliation: Batangas State University</p>
<p>E-mail: francismontalbo@ieee.org; francisjesmar.montalbo@g.batstate-u.edu.ph</p>
<p>Webpage: https://francismontalbo.github.io</p>

***Work is published in Machine Vision and Applications ISSN: 0932-8092.***

## CITE
**Montalbo, F.J.P, "Automated Diagnosis of Diverse Coffee Leaf Images through a Stage-Wise Aggregated Triple Deep Convolutional Neural Network," Machine Vision and Applications, vol. 33, no. 1, pp. 1-22, 2022. DOI: 10.1007/s00138-022-01277-y.**

Full paper access LINK:
https://link.springer.com/epdf/10.1007/s00138-022-01277-y?sharing_token=f8QRuXIgbHYoX4Ie9cAyhfe4RwlQNchNByi7wbcMAY5HInYw8kOdtP13cIiMaMWSKhoxFuB4Ar2PctzUJbjRUgfc00h_Az1pMSXF400fN15Dw-S1A-X04TfHDFsG_0niIhbxGnOZvEPVG3bLBq-zsheUavU3Nvox0AfknLjwDIA%3D

Original Paper Link:
https://link.springer.com/article/10.1007%2Fs00138-022-01277-y?fbclid=IwAR0nzpHsp4IaE42WywQ1xtXLJDOagDLTR4yf6_6YFgxiIiUI3mvr2Nke6Do

## Graphical Abstract

<b>Made with <a target=_blank href="https://draw.io">draw.io</a></b>
<img src="/graphical_abstract_diverse_coffee.jpg" alt="Stage Wise Aggregated Triple Deep Convolutional Neural Network for Coffee Leaf Diagnosis" width="1000">


## DATASETS

<p>The dataset used for this work came from the following works:</p>

**:warning: Please cite or credit their work when using it!** 

**RoCoLe** 
<p>Parraga-Alava, Jorge; Cusme, Kevin; Loor, Angélica; Santander, Esneider (2019), 
<b>“RoCoLe: A robusta coffee leaf images dataset ”</b>
<i>Mendeley Data</i>, V2, doi: <a target=_blank href="http://dx.doi.org/10.17632/c5yvn32dzg.2">10.17632/c5yvn32dzg.2</a></p>

Inclusion: 
- :heavy_check_mark: Healthy
- :heavy_check_mark: Coffee Leaf Rust (CLR)
- :heavy_check_mark: Red Spider Mites (RSM) 

**BrACoL** 
<p>Krohling, Renato; esgario, José; Ventura, Jose A. (2019),
<b>“BRACOL - A Brazilian Arabica Coffee Leaf images dataset to identification and quantification of coffee diseases and pests”</b>
<i>Mendeley Data</i>, V1, doi: <a target=_blank href="http://dx.doi.org/10.17632/yy2k5y8mxg.1">10.17632/yy2k5y8mxg.1</a></p>

<p>Esgario, J. G., Krohling, R. A., & Ventura, J. A. (2020) 
<b>"Deep learning for classification and severity estimation of coffee leaf biotic stress"</b>
<i>Computers and Electronics in Agriculture</i>
169, 105162. doi:<a href="https://doi.org/10.1016/j.compag.2019.105162">10.1016/j.compag.2019.105162</a></p>

Inclusion: 
- :heavy_check_mark: Healthy
- :heavy_check_mark: CLR
- :heavy_check_mark: Cercospora Leaf Spots (CLS)
- :heavy_check_mark: Phoma Leaf Spots (PLS)
- :heavy_check_mark: Coffee Leaf Miner (CLM)

**LiCoLe**
<p>Montalbo, Francis Jesmar Perez; Hernandez, Alexander Arsenio (2020) 
<b>"Classifying Barako coffee leaf diseases using deep convolutional models"</b>
<i>International Journal of Advances in Intelligent Informatics (IJAIN)</i>
[S.l.], v. 6, n. 2, p. 197-209, july 2020. ISSN 2548-3161. doi: <a href="https://doi.org/10.26555/ijain.v6i2.495">10.26555/ijain.v6i2.495</a></p>

<p>Montalbo, Francis Jesmar Perez
<b>"An Optimized Classification Model for Coffea Liberica Disease using Deep Convolutional Neural Networks"</b>
<i>n Proc. of the 2020 16th IEEE International Colloquium on Signal Processing & Its Applications (CSPA),</i> 
  Langkawi, Malaysia, 2020, pp. 213-218, doi: <a href="https://ieeexplore.ieee.org/document/9068683">10.1109/CSPA48992.2020.9068683</a>.</p>

Inclusion: 
- :heavy_check_mark: Healthy
- :heavy_check_mark: CLR
- :heavy_check_mark: Sooty Molds (SM)

**:heavy_exclamation_mark: For the readily prepared dataset used in this work refer to this link (OPTIONAL) 🠊 <a target=blank_ href="https://drive.google.com/drive/u/1/folders/1FyTnzfz0iLiiRMVWumEaoyFkX2YOHWz3">Google Drive Prepared Dataset<a/>** 
  
`PREPARED DATASET: (7 GB)`

***:warning: NOTE: The following credits for the datasets still goes to their appropriate owners and collectors.*** 
***:heavy_exclamation_mark: Please remember to cite their work when using their respective datasets.***

## Environment Setup

***:heavy_exclamation_mark: Make sure to create a new virtual environment preferably in Anaconda. Use Python 3.5.***

:arrow_right: https://www.anaconda.com/

:arrow_right: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

Activate and access the folder `swatdcnn/` with the included `requirements.txt` file.

**:warning: The SWAT-DCNN uses the tensorflow GPU. This may also require at least CUDA 10 and a cuDNN**

:warning: https://developer.nvidia.com/cuda-toolkit

https://developer.nvidia.com/rdp/cudnn-archive

Afterwards, simply enter the command in the conda CLI `pip install -r requirements.txt`

Dependencies include: 
- efficientnet==1.1.1
- colorama==0.4.3
- jupyter==1.0.0
- keras==2.2.5
- matplotlib
- numpy==1.16.2
- opencv-python==4.4.0.42
- pandas==0.25.3
- Pillow==7.2.0
- scikit-learn
- scikit-image
- scikit-plot
- scipy
- tensorflow-gpu==1.14.0

Once installed, you may either train the models individually with the `.ipynb` notebooks found in `swatdcnn/models/` inside the `stage-1`, `stage-2`, and `stage-3` folders or make use of the pre-trained weights.

The `swatdcnn/models/tdcnn/` files does not need to re-train. However, its a must to compile and aggregate the T-DCNN stages to produce its own respective weights needed by the entire SWAT-DCNN model.

## Pre-trained Weights ##

<p>The pre-trained weights are the plug and play weights that can be used to skip the training and compilation of models for the SWAT-DCNN (RECOMMENDED). :relaxed:</p>

**:heavy_exclamation_mark: For an immediate simulation without the hassle of going over the previous instructions, refer to this link.** 🠊 <a href="https://drive.google.com/drive/u/1/folders/1afxXTxP_i7nVFARRKWScUAH7FJZ2OPDh">Pre-Trained Weights</a>

`PRE-TRAINED WEIGHTS FILESIZE: (271 MB)`

The filenames must not be changed for the `.h5` files.

- `T-DCNN_stage-1.h5`
- `T-DCNN_stage-2.h5`
- `T-DCNN_stage-3.h5`

Make sure to extract the pre-trained weights in the given manner 🠊 `swatdcnn/weights/tdcnn/`

## How to use :octocat:

**:heavy_exclamation_mark: Training with the pre-trained weights (RECOMMENDED) :ok_hand:** 

1. Run the system by going to the swatdcnn folder with your created virtual environment activated and enter the command `python swatdcnn.py`

2. Follow through the given instructions and make sure to use the test sample from the provided `/test/` folder

**:warning: Training from scratch (May take long hours depending on your PC specs) :hand:**

1. Activate your created virtual environment and enter the main `swatdcnn/` folder.

2. Save the dataset folder downloaded from LINK inside the `swatdcnn/` as `swatdcnn/dataset/`

3. Open the `.ipynb` files from the `swatdcnn/models` folder and run the following in your preferred order. The `swatdcnn/models/tdcnn/` is saved for later.

4. Once all models from stage-1 to 3 are trained. You may now open the `swatdcnn/models/tdcnn/` folder to build the T-DCNN models.

5. After all T-DCNN models are built, you may now run the `swatdcnn.py` from the main `swatdcnn/` folder.

6. Follow through the given instructions and make sure to use the test sample from the provided `/test/` folder

**:heavy_exclamation_mark: In case of any problems, don't hesitate to contact me. I'll be happy to help.**

## Performance Results

<table style="width:100%">
  <tr>
    <th>Model</th>
    <th>Accuracy</th> 
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
  </tr>
  <tr>
    <td>SWAT-DCNN</td>
    <td><strong>95.98%</strong></td>
    <td><strong>96.27%</strong></td>
    <td><strong>95.98%</strong></td>
    <td><strong>95.91%</strong></td>
  </tr>
  <tr>
    <td>Base T-DCNN</td>
    <td>95.93%</td>
    <td>96.86%</td>
    <td>95.93%</td>
    <td>95.86%</td>
  </tr>
   <tr>
    <td>ResNet50V2</td>
    <td>92.82%</td>
    <td>92.51%</td>
    <td>92.82%</td>
    <td>92.50%</td>
  </tr>
   <tr>
    <td>Xception</td>
    <td>93.78%</td>
    <td>93.31%</td>
    <td>93.78%</td>
    <td>93.39%</td>
  </tr>
   <tr>
    <td>InceptionV3</td>
    <td>94.96%</td>
    <td>94.69%</td>
    <td>94.96%</td>
    <td>94.57%</td>
  </tr>
   <tr>
    <td>VGG19</td>
    <td>91.96%</td>
    <td>92.19%</td>
    <td>91.96%</td>
    <td>92.04%</td>
  </tr>
   <tr>
    <td>AlexNet</td>
    <td>81.35%</td>
    <td>80.06%</td>
    <td>81.35%</td>
    <td>80.08%</td>
  </tr>
   <tr>
    <td>LeNet-5</td>
    <td>71.17%</td>
    <td>68.35%</td>
    <td>71.17%</td>
    <td>69.68%</td>
  </tr>
</table>


## Citation :black_nib:

This GitHub repository serves as a support for the submitted publication article in Machine Vision and Applications

[Author Profile](https://scholar.google.com/citations?user=PV8dJDkAAAAJ&hl=en&oi=ao)




