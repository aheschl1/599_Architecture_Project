import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob 
import SimpleITK as sitk
from scipy.ndimage import label, sum, binary_dilation, find_objects
import numpy as np

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from multiprocessing.pool import ThreadPool as Pool
from threading import Thread
from time import sleep
import copy
from scipy import ndimage

final_kidney_path = "/home/student/andrew/Documents/Seg3D/nnunetv2/inference/kidney_good_results/fullres"
target_images = "/home/student/andrew/Documents/Dataset/SegData/nnUNet_raw/Dataset029_Tumour/imagesTr"
dilation = 3
stage = "MASS_ROI"
current_case = -1

assert stage in ["MASS_ROI", "KIDNEY_ROI"]

def get_case_name_from_number(case:int)->str:
    return '0'*(5-len(str(case))) + str(case)

class BBox:
    def __init__(self, slices) -> None:
        pass
        min_x = slices[0][1].start
        max_x = slices[0][1].stop
        min_y = slices[0][0].start
        max_y = slices[0][0].stop
        min_z = slices[0][2].start
        max_z = slices[0][2].stop

        self.coords = [[min_x,max_x],[min_y,max_y],[min_z,max_z]]

class PredictionHandler(FileSystemEventHandler):

    def __init__(self, folder:str) -> None:
        super().__init__()
        self.folder = folder

    @staticmethod
    def do_kidney_final_prediciton(kidney_prediciton:np.array, case:str, reference:sitk.Image)->None:
        global final_kidney_path
        kidney_prediciton[kidney_prediciton != 0] = 1
        kidney_prediciton = sitk.GetImageFromArray(kidney_prediciton)
        kidney_prediciton.CopyInformation(reference)
        sitk.WriteImage(kidney_prediciton, f"{final_kidney_path}/case_{case}.nii.gz")

    @staticmethod
    def get_bbox(mask:np.array)->list:
        labeled_image, c = ndimage.label(mask)
        # Get the height and width
        measurements = []
        for label in range(1, c+1):
            slices = find_objects(labeled_image == label)
            # Calculate the bounding box dimensions
            bbox = BBox(slices)
            measurements.append(bbox)
        return measurements
    
    @staticmethod
    def new_file_created(file:str):
        global target_images, stage

        print(f"Working on {file}")
        case = file.split('/')[-1].split('.')[0].split('_')[-1]

        target_path = f"{target_images}/case_{case}_0000.nii.gz"
        target = sitk.ReadImage(target_path)
        target_array = sitk.GetArrayFromImage(target)        
        #Target array is the image that we want to blank out. Target is the METADATA REFERENCE.

        prediction = sitk.GetArrayFromImage(sitk.ReadImage(file))
        if stage == "KIDNEY_ROI":
            prediction = PredictionHandler.remove_small_connected(prediction)
        else:
            _ = Thread(PredictionHandler.do_kidney_final_prediciton, args=(copy.deepcopy(prediction), case, target))
            _.start()
            prediction[prediction != 2] = 0
            prediction[prediction != 0] = 1

        dilated_prediction = PredictionHandler.get_dilated_mask(prediction)
        
        prediction = sitk.GetImageFromArray(prediction)
        prediction.CopyInformation(target)
        sitk.WriteImage(prediction, file)

        """
        If the stage is KIDNEY_ROI we now have:
            1. Removed the floating artifacts.
            2. Performed a dilation on the prediction and put it in a second copy.
            3. A list of bounding boxes around each kidney.
        If the stage is MASS_ROI we now have:
            1. A final kidney prediction saving to file
            2. A dilated version of the mass prediction
        """

        target_array *= dilated_prediction

        if stage == True: #fkjhdfakjfhk
            PredictionHandler.finish_mass_roi(target_array, target, target_path)
        else:
            PredictionHandler.finish_kidney_roi(target_array, target, dilated_prediction, case)

        print(f"Done case {case}.")
        return True

    @staticmethod
    def finish_kidney_roi(target_array, target:sitk.Image, dilated_prediction, case):
        global target_images, current_case
        labels_path = f"{target_images.replace('imagesTr', 'labelsTr')}/case_{case}.nii.gz"
        label = sitk.GetArrayFromImage(sitk.ReadImage(labels_path))

        print("Getting bboxes")
        bbox_list = PredictionHandler.get_bbox(dilated_prediction)
        print("Got em")
        print(len(bbox_list))
        for box in bbox_list:
            current_case += 1
            coords = box.coords
            new_case_name = get_case_name_from_number(current_case)
            print(target_array.shape)
            new_target_array = target_array[coords[0][0]:coords[0][1]+1, coords[1][0]:coords[1][1] + 1, coords[2][0]:coords[2][1]+1]
            new_label = label[coords[0][0]:coords[0][1]+1, coords[1][0]:coords[1][1] + 1, coords[2][0]:coords[2][1]+1]
            
            new_target_array = sitk.GetImageFromArray(new_target_array)
            new_label = sitk.GetImageFromArray(new_label)

            new_label.SetSpacing(target.GetSpacing())
            new_target_array.SetSpacing(target.GetSpacing())
            new_label.SetDirection(target.GetDirection())
            new_target_array.SetDirection(target.GetDirection())
            new_label.SetOrigin(target.GetOrigin())
            new_target_array.SetOrigin(target.GetOrigin())

            try:
                os.mkdir(f"{target_images}/new_images")
            except:
                pass

            try:
                os.mkdir(f"{target_images.replace('imagesTr', 'labelsTr')}/new_images")
            except:
                pass

            
            sitk.WriteImage(new_target_array, f"{target_images}/new_images/case_{new_case_name}_00000.nii.gz")
            sitk.WriteImage(new_target_array, f"{target_images.replace('imagesTr', 'labelsTr')}/new_images/case_{new_case_name}.nii.gz")


    @staticmethod 
    def finish_mass_roi(target_array, target, target_path):
        target_array = sitk.GetImageFromArray(target_array)
        target_array.CopyInformation(target)
        #Write the target_array image to the target_path.
        sitk.WriteImage(target_array, target_path)


    @staticmethod
    def get_dilated_mask(mask:np.array)->np.array:
        global dilation
        struct_element = np.ones((dilation, dilation, dilation), dtype=bool)
        dilated_mask = binary_dilation(mask, structure=struct_element)
        print("Done dilation")
        return dilated_mask

    @staticmethod
    def remove_small_connected(arr:np.array)->np.array:
        labeled_array, num_features = label(arr)
        volumes = np.zeros(num_features + 1)
        for i in range(1, num_features + 1):
            volume = sum(arr, labels=labeled_array, index=i)
            volumes[i] = volume

        threshold = np.sum(arr)//4
        mask_under_threshold = volumes < threshold
        labeled_array[mask_under_threshold[labeled_array]] = 0
        labeled_array[labeled_array!=0]=1
        return labeled_array

    def on_created(self, event):
        """Background task for when new cases are predicted."""
        if not event.is_directory:
            print(f"New file: {event.src_path}")
            thread = Thread(target=PredictionHandler.new_file_created, args=(event.src_path,))
            sleep(1)
            thread.start()
            # new_file_created(event.src_path)

if __name__ == '__main__':

    target = '/home/student/andrew/Documents/Seg3D/nnunetv2/inference/mass_results_rough'

    observer = Observer()
    event_handler = PredictionHandler(target)
    observer.schedule(event_handler, target, recursive=False)
    observer.start()

    for file in glob.glob(f'{target}/*.nii.gz'):
        PredictionHandler.new_file_created(file)

    observer.join()