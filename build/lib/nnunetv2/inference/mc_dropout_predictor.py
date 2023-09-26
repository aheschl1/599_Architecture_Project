from collections.abc import Callable, Iterable, Mapping
import inspect
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Any, Tuple, Union, List, Optional

import gc
import SimpleITK as sitk

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.ourTrainer import ourTrainer
from nnunetv2.inference.mc_dropout.mcdropout import MCDropoutBase

from nnunetv2.inference.calculating_prob import EntropyManager
from nnunetv2.inference.running_mean import PredictionManager
from nnunetv2.training.models.utils import ModuleStateController
import threading
from multiprocessing.pool import ThreadPool
import glob
from multiprocessing.pool import ThreadPool

class PredictionSave(threading.Thread):
    def __init__(self, path, image):
        super().__init__()
        self.path = path
        self.image = image

    def run(self):
        image = sitk.GetImageFromArray(self.image)
        sitk.WriteImage(image, self.path + ".nii.gz")

class CallThread(threading.Thread):
    def __init__(self, data, manager):
        super().__init__()
        self.data = data
        self.manager = manager
    def run(self):
        self.manager(self.data)


class DoneThread(threading.Thread):
    def __init__(self, data, manager):
        super().__init__()
        self.data = data
        self.manager = manager
    def run(self):
        self.manager.done(self.data)

class mcDropoutPredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_gpu: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
            super().__init__(tile_step_size,use_gaussian,use_mirroring,
                            perform_everything_on_gpu,device,verbose,
                            verbose_preprocessing,allow_tqdm)
            MCDropoutBase.activate()
            ModuleStateController.set_state(ModuleStateController.THREE_D)
            
    def initialize_from_trained_model_folder(self, 
                                             model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None]=None,
                                             model_path:str="",
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """

        print(model_training_output_dir)
        dataset_json = load_json(f"{'/'.join(model_training_output_dir.split('/')[0:-3])}/dataset.json")
        plans = load_json(f"{'/'.join(model_training_output_dir.split('/')[0:-3])}/plans.json")
        plans_manager = PlansManager(plans)

        parameters = []
        
        checkpoint = torch.load(join(model_training_output_dir, checkpoint_name), map_location=torch.device('cpu'))

        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
            'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')

        if trainer_class == ourTrainer:
            assert os.path.isfile(model_path), "Give a valid model path."
            network = trainer_class.build_network_architecture_static(plans_manager, dataset_json, configuration_manager,
                                                            num_input_channels, model_path, enable_deep_supervision=False)
        else:
            network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                            num_input_channels, enable_deep_supervision=False)
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('compiling network')
            self.network = torch.compile(self.network)

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0, iters=1):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        my_init_kwargs = {}
        for k in inspect.signature(self.predict_from_files).parameters.keys():
            my_init_kwargs[k] = locals()[k]
        my_init_kwargs = deepcopy(
            my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
        recursive_fix_for_json_export(my_init_kwargs)
        maybe_mkdir_p(output_folder)
        save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

        # we need these two if we want to do things with the predictions like for example apply postprocessing
        save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export, iters, output_folder )

    

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   _: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes,
                                   iters=1,
                                   output:str=""):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        for preprocessed in data_iterator:
            data = preprocessed['data']
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed['ofile']
            assert ofile is not None, "ofile shouldn't be None."
            if ofile is not None:
                print(f'\nPredicting {os.path.basename(ofile)}:')
            else:
                print(f'\nPredicting image of shape {data.shape}:')

            print(f'perform_everything_on_gpu: {self.perform_everything_on_gpu}')

            properties = preprocessed['data_properites']
            
            # Initialize a list to hold the entropy sum for each class
            entropy_manager = EntropyManager(output)
            threads_container = []
            for w in range(iters):
                print(f"Working on iteration {w +1}/{iters}")
                prediction = self.predict_logits_from_preprocessed_data(data)

                seg, prob = convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction, self.plans_manager, self.configuration_manager, self.label_manager, properties,
                    return_probabilities=True, numpy_type=False
                )

                threads_container.append(EntropyNewProbThread(prob, entropy_manager))
                threads_container[-1].start()

                if w == 0:
                    threads_container.append(PredictionSave(ofile, seg))
                    threads_container[-1].start()

            for thread in threads_container:
                thread.join()
            
            t = EntropyDoneThread(ofile.split('/')[-1].split('_')[1], entropy_manager)
            t.start()
                            
                seg, prob = convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction, self.plans_manager, self.configuration_manager, self.label_manager, properties,
                    return_probabilities=True, numpy_type=False
                )

                threads_container.append(CallThread(prob, entropy_manager))
                threads_container[-1].start()

                threads_container.append(CallThread(prob, prediction_manager))
                threads_container[-1].start()

                if w == 0:
                    threads_container.append(PredictionSave(ofile, seg))
                    threads_container[-1].start()

            for thread in threads_container:
                thread.join()
            
            t = DoneThread(ofile.split('/')[-1].split('_')[1], entropy_manager)
            t.start()
            thread_queue.append(t)

            t = DoneThread(ofile.split('/')[-1].split('_')[1], prediction_manager)
            t.start()
            thread_queue.append(t)
                            
        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)

        for thread in thread_queue:
            thread.join()

        return True 

class PredictionCombiner:
    def __init__(self, kidney_path:str, masses_path:str, tumour_path:str, threshold:float = 0.9) -> None:
        self.kidney_path = kidney_path
        self.masses_path = masses_path
        self.tumour_path = tumour_path
        self.threshold = threshold

    def combine(self, output_dir:str):
        cases = glob.glob(f"{self.kidney_path}/*.nii.gz")
        cases = [file.split('/')[-1].split('.')[0].split('_')[-1] for file in cases]
        #Now we have a list of case ids to work with
        for case in cases:
            final_prediction = self.predict(case)
            sitk.SaveImage(final_prediction, f"{output_dir}/case_{case}.nii.gz")

    def get_data(self, case:str):
        entropy_path_kidney = f"{self.kidney_path}/entropycase_{case}/case{case}_channel0.nii.gz"
        entropy_path_masses = f"{self.masses_path}/entropycase_{case}/case{case}_channel0.nii.gz"
        entropy_path_tumour = f"{self.tumour_path}/entropycase_{case}/case{case}_channel0.nii.gz"

        entropy_kidney = sitk.GetArrayFromImage(sitk.ReadImage(entropy_path_kidney))
        entropy_masses = sitk.GetArrayFromImage(sitk.ReadImage(entropy_path_masses))
        entropy_tumour = sitk.GetArrayFromImage(sitk.ReadImage(entropy_path_tumour))
        
        kidney_above = np.argwhere(entropy_kidney > self.threshold)
        mass_above = np.argwhere(entropy_masses > self.threshold)
        tumour_above = np.argwhere(entropy_tumour > self.threshold)            

        mask_path_kidney = f"{self.kidney_path}/predictioncase_{case}/case_{case}.nii.gz"
        mask_path_masses = f"{self.kidney_path}/predictioncase_{case}/case_{case}.nii.gz"
        mask_path_tumour = f"{self.kidney_path}/predictioncase_{case}/case_{case}.nii.gz"

        kidney_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path_kidney))
        mass_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path_masses))
        tumour_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path_tumour))

        for (x,y,z) in kidney_above:
            kidney_mask[x,y,z] = 0
        for (x,y,z) in mass_above:
            mass_mask[x,y,z] = 0
        for (x,y,z) in tumour_above:
            tumour_mask[x,y,z] = 0

        return (kidney_mask, entropy_kidney),(mass_mask, entropy_masses),(tumour_mask,entropy_tumour)

    def predict(self, case:str)->sitk.Image:
        (kidney_seg, kidney_entropy), (masses_seg, masses_entropy), (tumour_seg, tumour_entropy) = self.get_data(case)
        
        conflicts = np.argwhere(kidney_seg != masses_seg)
        kidney_seg[masses_seg + kidney_seg == 2] = 2
        for (x, y, z) in conflicts:
            #Mass predicted, but kidney not. Take most confident.
            if kidney_seg[x, y, z] == 0:
                if kidney_entropy[x, y, z] > masses_entropy[x, y, z]:
                    kidney_seg[x, y, z] = 2
        
        #Now kidney_seg should be the combined kidney masses. Now combine with tumour
        kidney_seg[tumour_seg + kidney_seg == 3] = 3
        one_and_one = np.argmax((tumour_seg == 1) & (kidney_seg == 1))
        for (x, y, z) in one_and_one:
            if kidney_entropy[x, y, z] > tumour_entropy[x, y, z]:
                kidney_seg[x, y, z] = 3
        
        two_and_0 = np.argmax((tumour_seg == 2) & (kidney_seg == 0))
        for (x, y, z) in two_and_0:
            if masses_entropy[x, y, z] > tumour_entropy[x, y, z]:
                kidney_seg[x, y, z] = 3

class PredictorTask(threading.Thread):
    def __init__(self, predictor:mcDropoutPredictor, model_folder:str, input_files:str, output:str, weights:str='checkpoint_best.pth'):
        super().__init__()
        self.predictor = predictor
        self.model_folder = model_folder
        self.weights = weights
        self.input_files = input_files
        self.output = output

    def run(self):
        self.predictor.initialize_from_trained_model_folder(self.model_folder, checkpoint_name=self.weights)
        self.predictor.predict_from_files(self.input_files,
                                 self.output,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=4, num_processes_segmentation_export=4,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0, iters=10)



if __name__ == '__main__':
    # predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw

    #=============================================START KIDNEY==========================================
    predictor_kidney = mcDropoutPredictor(
        tile_step_size=1,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda:0'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True)
    
    kidney_task = PredictorTask(
        predictor_kidney, 
        "/home/student/andrew/Documents/Dataset/SegData/nnUNet_results/Dataset027_Kidney/nnUNetTrainer__nnUNetPlans__3d_fullres_05/fold_0/2023_7_9/9039",
        join(nnUNet_raw, 'Dataset027_Kidney/imagesTr'),
        join(nnUNet_results, 'Dataset027_Kidney/predictions'),
    )
    kidney_task.start()
    #=============================================START MASSES==========================================
    predictor_masses = mcDropoutPredictor(
        tile_step_size=1,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda:1'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True)
    masses_task = PredictorTask(
        predictor_masses, 
        "/home/student/andrew/Documents/Dataset/SegData/nnUNet_results/Dataset028_Masses/nnUNetTrainer__nnUNetPlans__3d_fullres_05/fold_0/2023_7_9/9625",
        join(nnUNet_raw, 'Dataset028_Masses/imagesTr'),
        join(nnUNet_results, 'Dataset028_Masses/predictions'),
    )
    masses_task.start()

    #=============================================WAIT FOR GPU==========================================
    kidney_task.join()
    masses_task.join()
    #=============================================START TUMOUR==========================================
    predictor_tumour = mcDropoutPredictor(
        tile_step_size=1,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda:0'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True)
    masses_task = PredictorTask(
        predictor_masses, 
        "/home/student/andrew/Documents/Dataset/SegData/nnUNet_results/Dataset029_Tumour/nnUNetTrainer__nnUNetPlans__3d_fullres_05/fold_0/2023_7_9/9430",
        join(nnUNet_raw, 'Dataset029_Tumour/imagesTr'),
        join(nnUNet_results, 'Dataset029_Tumour/predictions'),
    )
    #=============================================WAIT==========================================
    masses_task.start()
    masses_task.join()

    #Now we have entropy maps and predictions for each model. Combining them will follow
    final_prediction_output = f"{nnUNet_results}/final_predicitons"
    os.makedirs(final_prediction_output, exist_ok=True)

    combiner = PredictionCombiner(
        join(nnUNet_results, 'Dataset027_Kidney/predictions'), 
        join(nnUNet_results, 'Dataset028_Masses/predictions'), 
        join(nnUNet_results, 'Dataset029_Tumour/predictions')
    )
    predictor.predict_from_files(join(nnUNet_raw, 'Dataset021_All/imagesTr'),
                                 output,
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0, iters=10)
                                 