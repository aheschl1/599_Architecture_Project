import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob 
import SimpleITK as sitk
import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from multiprocessing.pool import ThreadPool as Pool
from threading import Thread
from time import sleep
import copy
from scipy import ndimage
from utils import ROOT_DIR, NNUNET_RAW, NNUNET_RESULTS, get_case_name_from_number, get_configurations, ConfigKeys
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

"""
THIS FILE MAY BE USELESS. Tumour predictions will be made with a single step. We can combine from there, or maybe here.

It is indeed useless...... :(
"""

def predict_data(config:dict, prediction_path:str):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
        )
    
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=f"{NNUNET_RESULTS}/{config[ConfigKeys.DATASET_TUMOUR.value]}/nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=[0],
        checkpoint_name='checkpoint_best.pth',
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder=f"{ROOT_DIR}/{config[ConfigKeys.INFERENCE.value]}/tumour_inference",
        output_folder_or_list_of_truncated_output_files=prediction_path,
        save_probabilities=False, overwrite=False,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
    )

if __name__ == '__main__':

    configs = get_configurations()

    do_predictions = True
    prediction_path = f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/tumour_predictions"

    if do_predictions:
        predict_process = Thread(
            target = predict_data,
            args = (configs,prediction_path)
        )
        predict_process.start()

        predict_process.join()
    # observer = Observer()
    # event_handler = TumourPredictionHandler(prediction_path, f"{ROOT_DIR}/{configs[ConfigKeys.TUMOUR_RESULTS]}")
    # observer.schedule(event_handler, prediction_path, recursive=False)
    # observer.start()

    # already_ready_predictions = glob.glob(f'{prediction_path}/*.nii.gz')
    # for file in already_ready_predictions:
    #     event_handler.new_file_created(file)

    # observer.join()