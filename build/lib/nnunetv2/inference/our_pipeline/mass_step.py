import torch
import torch.nn.functional as F
import glob
import SimpleITK as sitk
from scipy.ndimage import label, sum, binary_dilation
import numpy as np

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from threading import Thread
from time import sleep
from utils import ROOT_DIR, NNUNET_RAW, NNUNET_RESULTS, get_configurations, ConfigKeys
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch.nn as nn

"""
Coming here, we should have a fully generated kidney region dataset.
"""


class MassPredictionHandler(FileSystemEventHandler):
    """
    Manages the creation of kidney ROI dataset from the prediction of kidney region.
    """

    def __init__(self, observer: Observer, folder: str, tumour_dataset_images: str, config: dict,
                 total_expected_cases: int, dilation_device:str = "cuda:0") -> None:
        super().__init__()
        self.folder = folder
        self.tumour_dataset_images = tumour_dataset_images
        self.config = config
        self.observer = observer
        self.total_expected_cases = total_expected_cases
        self.processed_cases = 0
        self.dilation_device = dilation_device

    def new_file_created(self, file_path: str):
        """
        A new mass ROI has been predicted.
        """
        print(f"Working on {file_path}")
        case = file_path.split('/')[-1].split('.')[0].split('_')[-1]

        target_path = f"/home/andrew.heschl/Documents/Seg3D/nnunetv2/inference/our_pipeline/inference/mass_inference/case_{case}_0000.nii.gz"
        target = sitk.ReadImage(target_path)
        target_array = sitk.GetArrayFromImage(target)
        # Target array is the image that we want to blank out. Target is the METADATA REFERENCE.

        prediction = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        dilated_prediction = self._get_dilated_mask(prediction)

        """
        We now have:
            1. Performed a dilation on the prediction and put it in a second copy (dilated_prediciton).
        """
        self._finish_mass_roi(target_array, target, dilated_prediction, case)
        print(f"Done case {case}.")
        self.processed_cases += 1
        if self.processed_cases == self.total_expected_cases:
            self.observer.stop()
            return True
        return False

    def _finish_mass_roi(self, target_array: np.array, target: sitk.Image, dilated_prediction: np.array, case: str):
        """
        This method should be in charge of cropping out ROI.
        """
        target_array *= dilated_prediction
        new_raw_image = sitk.GetImageFromArray(target_array)
        new_raw_image.CopyInformation(target)

        sitk.WriteImage(new_raw_image, f"{self.tumour_dataset_images}/case_{case}_0000.nii.gz")

    def _get_dilated_mask(self, mask: np.array) -> np.array:
        """
        Given a numpy array, returns a new one which has been dilated with a cubed kernel of the size in config file.
        """
        #cpu version
        # dilation = self.config[ConfigKeys.KIDNEY_DILATION.value]
        # struct_element = np.ones((dilation, dilation, dilation), dtype=bool)
        # dilated_mask = binary_dilation(mask, structure=struct_element)
        # print("Done dilation")
        # return dilated_mask

        #torch gpu version
        mask = torch.Tensor(mask)
        dtype, mask = mask.dtype, mask.float().unsqueeze(0).to(self.dilation_device)
        # Create a maxpool3d layer
        dilation = self.config[ConfigKeys.MASS_DILATION.value]
        maxpool3d = nn.MaxPool3d(dilation, stride=1, padding=dilation // 2)
        dilated_segmentation = maxpool3d(mask)
        return np.array((dilated_segmentation > 0.5).type(dtype).squeeze().to('cpu'))

    def on_created(self, event):
        """
        Background task for when new cases are predicted.
        """
        if not event.is_directory:
            print(f"New file: {event.src_path}")
            thread = Thread(target=self.new_file_created, args=(event.src_path,))
            sleep(1)
            thread.start()


def predict_data(config: dict, prediction_path: str):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda:0'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=
        f"{NNUNET_RESULTS}/{config[ConfigKeys.DATASET_MASS.value]}/nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    predictor.predict_from_files(
        list_of_lists_or_source_folder=f"{ROOT_DIR}/{config[ConfigKeys.INFERENCE.value]}/mass_inference",
        output_folder_or_list_of_truncated_output_files=prediction_path,
        save_probabilities=False, overwrite=False,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
    )


if __name__ == '__main__':

    configs = get_configurations()

    total_expected_cases = len(glob.glob(f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/mass_inference/*.nii.gz"))

    do_predictions = True
    prediction_path = f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/mass_predictions"

    if do_predictions:
        predict_process = Thread(
            target=predict_data,
            args=(configs, prediction_path)
        )
        predict_process.start()

    observer = Observer()
    event_handler = MassPredictionHandler(
        observer,
        folder=prediction_path,
        tumour_dataset_images=f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/tumour_inference",
        config=configs,
        total_expected_cases=total_expected_cases
    )

    observer.schedule(event_handler, prediction_path, recursive=False)
    observer.start()

    already_ready_predictions = glob.glob(f'{prediction_path}/*.nii.gz')
    for file in already_ready_predictions:
        event_handler.new_file_created(file)

    observer.join()
