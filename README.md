# Collateral Scoring
The data and code for the paper "A Dual-Branch Hybrid Network with Bilateral-Difference Awareness for Collateral Scoring on CT Angiography of Acute Ischemic Stroke Patients" submitted to Computer Methods and Programs in Biomedicine 2024. <br />

## Requirements
CUDA 11.7<br />
Python 3.9.19<br /> 
Pytorch 2.2.1<br />
Torchvision 0.17.1<br />
SimpleITK 2.4.0 <br />
einops 0.8.0 <br />
matplotlib 3.9.2 <br />
timm 1.0.9 <br />
scikit_learn 1.5.2 <br />
opencv_python 4.10.0.84 <br />
pandas 2.2.3 <br />
numpy 2.1.1 <br />

## Getting Started

### Installation
#### Step1: Clone the Collateral repository
To get started, first clone the Collateral repository and navigate to the project directory:

```
git clone https://github.com/09Echo/Collateral.git
cd Collateral
```

#### Step2: Environment Setup
Create and activate a new conda environment
```
conda create -n collateral python=3.9.19
conda activate collateral
```

Install Dependencies
```
pip install -r requirements.txt
```

## Collateral Scoring on CT Angiography
### Skull-stripping
  
After converting the DICOM files to NIfTI format, perform skull stripping according to the instructions at https://github.com/WuChanada/StripSkullCT.  <br />

### Training  
```
python train.py --save-path <Path> --num-classes <Classes>
```
Parameter Description：  
* --save-path <Path>: Model save path, please specify a valid directory
* --num-classes <Classes>: Classification category, please specify a positive integer, such as 3

### Testing  

### Reproduction details and codes
During reproduction, for the CNN-based methods, Transformer-based methods, and Hybrid-CNN-Transformer-based methods. All of these methods can be found at [[Baseline]](./baseline).  <br />
Note that for all compared methods, to perform fair comparisons, we used the same data split and five-fold cross-validation.  <br />



