import SimpleITK as sitk
import numpy as np
from utils import get_configurations, ConfigKeys, NNUNET_RAW
import os
import shutil
import glob

config = get_configurations()
dataset_path = f"{NNUNET_RAW}/{config[ConfigKeys.DATASET_TUMOUR.value]}"

for_removal = [f"000{x}" for x in [28, 35, 38, 41, 50, 43, 44, 46, 47, 51, 66, 73, 77, 95]]
for_removal.append("00102")


try:
    os.makedirs(f"{dataset_path}/black_images")
except:
    pass

samples = glob.glob(f"{dataset_path}/imagesTr/*.nii.gz")

for sample in samples:
    case = sample.split('/')[-1].split('.')[0].split('_')[-1]
    image = sitk.ReadImage(sample)
    image_array = sitk.GetArrayFromImage(image)

    if np.sum(image_array) == 0 or case in for_removal:
        shutil.move(sample, f"{dataset_path}/black_images/{sample.split('/')[-1]}")
        shutil.move(sample.replace('imagesTr', 'labelsTr').replace('_0000.nii.gz', '.nii.gz'), f"{dataset_path}/black_images/{sample.split('/')[-1]}")