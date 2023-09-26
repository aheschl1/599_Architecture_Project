import SimpleITK as sitk
import numpy as np
import operator
import glob

def pad(img:np.array, target_size:tuple)->np.array:
    container = np.zeros(target_size)
    print(target_size, img.shape)
    container[
        (img.shape[0]-target_size[0])//2:(img.shape[0]-target_size[0])//2+img.shape[0]+1,
        (img.shape[1]-target_size[1])//2:(img.shape[1]-target_size[1])//2+img.shape[1]+1,
        (img.shape[2]-target_size[2])//2:(img.shape[2]-target_size[2])//2+img.shape[2]+1,
    ] = img
    return container

def cropND(img:np.array, target_size)->np.array:

    def get_coords(x, x_t):
        x_start, x_end = 0, x
        if x > x_t:
            if (x_t - x) % 2 == 1:
                x_end -= 1

            difference = (x_t - x)//2
            x_start += difference
            x_end -= difference
            
        return x_start, x_end
    
    x, y, z = img.shape
    x_t, y_t, z_t = target_size

    x_start, x_end = get_coords(x, x_t)
    y_start, y_end = get_coords(y, y_t)
    z_start, z_end = get_coords(z, z_t)

    return img[
        x_start:x_end,
        y_start:y_end,
        z_start:z_end
    ]

import os

input = "/home/student/andrew/Documents/predicted_mass_rough"
output = f"{input}/kits_format"

os.makedirs(output, exist_ok=True)

for path in glob.glob(f"{input}/*.nii.gz"):
    seg = sitk.ReadImage(path)

    casse_name = path.split('/')[-1].split('_')[1].split('.')[0]
    image = f"/data/Datasets/KITS23/kits23/dataset/case_{casse_name}/imaging.nii.gz"

    image = sitk.ReadImage(image)

    seg_array = cropND(sitk.GetArrayFromImage(seg), image.GetSize())
    seg_array = pad(seg_array, image.GetSize())

    seg = sitk.GetImageFromArray(seg_array)

    seg.SetSpacing(image.GetSpacing())
    seg.SetDirection(image.GetDirection())
    seg.SetOrigin(image.GetOrigin())

    sitk.WriteImage(seg, f'{output}/case_{casse_name}.nii.gz')