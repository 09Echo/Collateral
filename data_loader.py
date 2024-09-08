import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from monai.transforms import (
    OneOf,
    LoadImage,
    Transpose,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandGaussianSharpen,
    RandHistogramShift,
    NormalizeIntensity,
    Rand2DElastic,
    RandAffine,
    Zoom,
    Compose,
)
import cv2
dataset_path = '/home/hubin/Collateral/Datasets/MIP_25p_Rotate_Crop_Pad'
data_info_path = "/home/hubin/Collateral/ProveIt_Select_Sheet0505_select.xlsx"


class ProVe():
    def __init__(self, fold=0, mode='train', flag=0):
        modalities = ['mCTA1_brain_mca_max.nii.gz']
        self.mode = mode
        self.fold = fold
        self.split = []
        df = pd.read_excel(data_info_path)[['Prove-it ID', 'Collaterals (1:Good, 2:inter, 3:poor)']]
        df.columns = ['id', 'label']
        data_list = []
        label_list = []
        id_list = []
        for idx, row in df.iterrows():
            img_id = row['id']
            label = int(row['label']) - 1
            img_path = []
            for m in modalities:
                data_path = os.path.join(dataset_path, img_id, m)
                assert os.path.isfile(data_path), data_path
                img_path.append(data_path)
            data_list.append(img_path)
            label_list.append(label)
            id_list.append(img_id)
        data = np.array(data_list)
        label = np.array(label_list)
        id = np.array(id_list)
        if flag == 1:
            label = np.where(label > 1, 1, 0)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=99)
        # rus = RandomUnderSampler(random_state=42, sampling_strategy={0: 80})
        # ros = RandomOverSampler(random_state=42, sampling_strategy={1: 70, 2: 55})
        ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
        for train_index, test_index in skf.split(data, label):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            z_id = id[test_index]
            self.split.append({'train': {'path': X_train, 'label': y_train},
                               'test': {'path': X_test, 'label': y_test, 'id': z_id}})
        for split in self.split:
            # X, y = rus.fit_resample(split['train']['path'], split['train']['label'])
            # split['train']['path'], split['train']['label'] = X, y
            X, y = ros.fit_resample(split['train']['path'], split['train']['label'])
            split['train']['path'], split['train']['label'] = X, y
        print()
        if mode == 'train':
            # self.transform = Compose([
            #     # Transpose(indices=(2, 0, 1)),
            #     NormalizeIntensity(),
            #     RandGaussianNoise(prob=0.1),
            #
            #     RandGaussianSharpen(prob=0.1),
            #     RandGaussianSmooth(prob=0.1),
            #     #
            #     RandHistogramShift(prob=0.1),
            #     Rand2DElastic(prob=0.2, spacing=(20, 20), magnitude_range=(1, 2)),
            #     RandAffine(rotate_range=np.pi / 12, translate_range=20, prob=0.5),
            # ])
            self.transform = Compose([
                # Transpose(indices=(2, 0, 1)),
                NormalizeIntensity(),
            ])
        else:
            self.transform = Compose([
                # Transpose(indices=(2, 0, 1)),
                NormalizeIntensity(),
            ])

    def __getitem__(self, item):
        img_info = self.split[self.fold][self.mode]
        img_list = img_info['path'][item]
        itk = sitk.ReadImage(img_list)
        # arr = sitk.GetArrayFromImage(itk).squeeze()
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # arr = min_max_scaler.fit_transform(arr)
        # label = img_info['label'][item]
        # return arr[None].astype(float), label
        arr = sitk.GetArrayFromImage(itk).astype(float)
        arr = cv2.resize(arr.squeeze(), (512, 512), interpolation=cv2.INTER_AREA)
        arr = arr[None]
        arr = self.transform(arr)
        # min_max_scaler = preprocessing.StandardScaler()
        # arr = min_max_scaler.fit_transform(arr.squeeze())
        # arr = arr[None]
        label = img_info['label'][item]
        # label = label[None]
        if self.mode == 'test':
            id = img_info['id'][item]
            return arr.astype(float), label, id
        else:
            return arr.astype(float), label
        # return arr.astype(float), label

    def __len__(self):
        return len(self.split[self.fold][self.mode]['path'])


if __name__ == '__main__':
    dataset = ProVe(flag=0, mode='test')
    print(dataset[0][0].shape)
    plt.imshow(dataset[0][0][0], cmap='gray')
    plt.show()
    print('ok')
