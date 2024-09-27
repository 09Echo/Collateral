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

## Usage

### 0. Installation
* Install our modified nnUNet as below
  
```
git clone https://github.com/GitHub-TXZ/AISCT-SAM.git
cd AISCT-SAM
pip install -e .

```
### 1 Acute Ischemic Stroke Dataset (AISD)
### 1.1 Dataset access
AISD dataset can be downloaded from (https://github.com/griffinliang/aisd).  <br />

### 1.2 Skull-stripping
After converting the DICOM files of the AISD dataset to NIfTI format, perform skull stripping according to the instructions at https://github.com/WuChanada/StripSkullCT.  <br />

### 1.3 Flip-Registration
Then, perform flip registration according to ./myscripts/Registration. Finally, organize the dataset in the nnUNet-expected format according to the code in nnUNet/nnunet/dataset_conversion.  <br />

### 1.4 Pre-processing
Some compared methods use the same pre-processing steps as nnUNet. The documentation of the pre-processing can be found at [[DOC]](./nnUNet/documentation) <br />

### 1.5 Training
conda activate <YOUR ENV NAME>
Simply run the following in your command line:
* Run `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -dataset_name_or_id TASK_ID -model_name AIS_SAM -ex_name Ex1@b_2_p_20_256_256_s_3.0_0.4375_0.4375` for training.  <br />


### 1.6 Testing 
* Run `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -dataset_name_or_id TASK_ID -model_name AIS_SAM -ex_name Ex1@b_2_p_20_256_256_s_3.0_0.4375_0.4375 --val` for testing.  <br />


### 2.1 Pre-trained model
The pre-trained model of AISD dataset can be downloaded from [[Baidu YUN]](https://pan.baidu.com/s/1m2YjNKDkr1bwF7DtIH1Tdw) with the password "14av".  <br />


### 2.2 Reproduction details and codes 
 During reproduction, for the CNN-based methods, Transformer-based methods, Hybrid-CNN-Transformer-based methods, Mamba-based mehtods. We integrated them into the nnUNet framework. All of these 3D methods can be found at [[DOC]](./Three_d).  <br />

For the AIS-Specific methods and SAM-based methods. We endeavored to implement them using our AIS datasets.our reproduced codes. The links of their open-source codes are listed as follows: <br />

[[Kuang et al.]](https://github.com/hulinkuang/Cl-SegNet) </br>
[[UNet-RF]](https://github.com/WuChanada/Acute-ischemic-lesion-segmentation-in-NCCT) </br>
[[ADN]](https://github.com/nihaomiao/MICCAI22_ADN) </br>
[[SAM-Med2D]](https://github.com/OpenGVLab/SAM-Med2D) </br>
[[SAM]](https://github.com/facebookresearch/segment-anything) </br>
[[SAM-Med3D]](https://github.com/uni-medical/SAM-Med3D) </br>
[[MedSAM]](https://github.com/bowang-lab/MedSAM) </br>
[[MSA]](https://github.com/KidsWithTokens/Medical-SAM-Adapter) </br>
[[3DSAM Adapter]](https://github.com/med-air/3DSAM-adapter) </br>
[[SAMed]](https://github.com/hitachinsk/SAMed) </br>


Note that for all compared methods, to perform fair comparisons, we used he same data split and all metrics were computed at the 3D image level.  <br />


## Acknowledgements
Part of codes are reused from the nnU-Net, thanks to Fabian Isensee for the owesome codes of nnUNet. And we express our sincerest gratitude to all the awesome open-source code that we have used in our work. <br />
