import torch
import torch.nn.functional as F
import glob
import SimpleITK as sitk
from scipy.ndimage import label, sum, binary_dilation
import numpy as np
import os

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from threading import Thread
from time import sleep
from utils import ROOT_DIR, NNUNET_RAW, NNUNET_RESULTS, get_configurations, ConfigKeys, remove_small_connected
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from chunk_management.image_chunk_management import ImageCropManagement
import torch.nn as nn

"""
TODO change the configs label lookup path to be the one to the masses labels.
Test.
"""


class KidneyPredictionHandler(FileSystemEventHandler):
    """
    Manages the creation of kidney ROI dataset from the prediction of kidney region.
    """

    def __init__(
            self,
            folder: str,
            mass_dataset_images: str,
            mass_dataset_labels: str,
            tumour_dataset_labels: str,
            config: dict,
            observer: Observer,
            total_expected_cases=489,
            dilation_device: str = 'cuda'
    ) -> None:
        super().__init__()
        self.folder = folder
        self.mass_dataset_images = mass_dataset_images
        self.config = config
        self.crop_management = None
        self.total_expected_cases = total_expected_cases
        self.processed_cases = 0
        self.observer = observer
        self.mass_dataset_labels = mass_dataset_labels
        self.tumour_dataset_labels = tumour_dataset_labels
        self.dilation_device = dilation_device

    def _setup_crop_management(self, reference: sitk.Image) -> None:
        self.crop_management = ImageCropManagement(
            self.mass_dataset_images,
            [self.mass_dataset_labels, self.tumour_dataset_labels],
            "case_*_0000",
            "case_*",
            reference.GetSpacing(),
            reference.GetDirection(),
            reference.GetOrigin(),
        )

    def new_file_created(self, file_path: str) -> bool:
        """
        A new kidney ROI has been predicted. Collects the case name from the file, and then overwrites
        the mass dataset with the new raw images.
        """
        if('json' in file_path):
            return
        print(f"Working on {file_path}")
        case = file_path.split("/")[-1].split(".")[0].split("_")[-1]
        # target path is the file where we grab the raw image to put in the mass dataset.
        target_path = f"/home/student/andrew/Documents/Seg3D/nnunetv2/inference/our_pipeline/inference/raw/case_{case}_0000.nii.gz"
        target = sitk.ReadImage(target_path)
        if self.crop_management is None:
            self._setup_crop_management(target)

        self.crop_management.direction = target.GetDirection()
        self.crop_management.origin = target.GetOrigin()
        self.crop_management.spacing = target.GetSpacing()

        target_array = sitk.GetArrayFromImage(target)
        # Target array is the image that we want to blank out. Target is the METADATA REFERENCE.

        prediction = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        prediction = remove_small_connected(prediction)
        dilated_prediction = self._get_dilated_mask(prediction)

        # Re-write the original prediction which has had the small chunks removed, but was not dilated.
        # DO NOT OVERWRITE THE DILATED VERSION!!!!!!!!!
        prediction = sitk.GetImageFromArray(prediction)
        # prediction.CopyInformation(target)
        prediction.SetSpacing(target.GetSpacing())
        prediction.SetDirection(target.GetDirection())
        prediction.SetOrigin(target.GetOrigin())

        sitk.WriteImage(prediction, file_path)

        """
        We now have:
            1. Removed the floating artifacts, and rewrote the prediction.
            2. Performed a dilation on the prediction and put it in a second copy (dilated_prediciton).
        """
        self._finish_kidney_roi(target_array, dilated_prediction, case)

        print(f"Done case {case}.")
        self.processed_cases += 1
        if self.processed_cases == self.total_expected_cases:
            self.observer.stop()
            return True
        return False

    def _finish_kidney_roi(
            self, target_array: np.array, dilated_prediction: np.array, case: str
    ):
        """
        This method should be in charge of cropping out ROI.
        """
        target_array *= dilated_prediction
        # new_raw_image = sitk.GetImageFromArray(target_array)
        # new_raw_image.CopyInformation(target)

        # sitk.WriteImage(new_raw_image, f"{self.mass_dataset_images}/case_{case}_00000.nii.gz")
        if self.mass_dataset_labels != None:
            target_label = sitk.ReadImage(
                f"{self.mass_dataset_labels}/uncropped_labels/case_{case}.nii.gz"
            )
            target_label2 = sitk.ReadImage(
                f"{self.tumour_dataset_labels}/uncropped_labels/case_{case}.nii.gz"
            )
        # TODO uncomment below for file cleanup
        # os.remove(f"{self.mass_dataset_labels}/uncropped_labels/case_{case}.nii.gz")
        self.crop_management.new_image(
            basis=dilated_prediction,
            image=target_array,
            labels=[sitk.GetArrayFromImage(target_label), sitk.GetArrayFromImage(target_label2)] if self.mass_dataset_labels != None else [],
            parent_case=case
        )

    def _get_dilated_mask(self, mask: np.array) -> np.array:
        """
        Given a numpy array, returns a new one which has been dilated with a cubed kernel of the size in config file.
        """
        #cuda version
        # dilation = self.config[ConfigKeys.KIDNEY_DILATION.value]
        # struct_element = np.ones((dilation, dilation, dilation), dtype=bool)
        # dilated_mask = binary_dilation(mask, structure=struct_element)
        # print("Done dilation")
        # return dilated_mask

        #torch gpu version
        mask = torch.Tensor(mask)
        dtype, mask = mask.dtype, mask.float().unsqueeze(0).to(self.dilation_device)
        # Create a maxpool3d layer
        dilation = self.config[ConfigKeys.KIDNEY_DILATION.value]
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
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=
        f"{NNUNET_RESULTS}/{config[ConfigKeys.DATASET_KIDNEY.value]}/nnUNetTrainer__nnUNetPlans__3d_fullres",
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )
    print(f"{ROOT_DIR}/{config[ConfigKeys.INFERENCE.value]}/raw")
    predictor.predict_from_files(
        list_of_lists_or_source_folder=f"{ROOT_DIR}/{config[ConfigKeys.INFERENCE.value]}/raw",
        output_folder_or_list_of_truncated_output_files=prediction_path,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=16,
        num_processes_segmentation_export=16,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


if __name__ == "__main__":
    configs = get_configurations()
    # print(ROOT_DIR)

    do_predictions = False
    prediction_path = f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/kidney_predictions"

    if do_predictions:
        predict_process = Thread(target=predict_data, args=(configs, prediction_path))
        predict_process.start()

    observer = Observer()
    # event_handler = KidneyPredictionHandler(
    #     folder=prediction_path,
    #     mass_dataset_images=f"{NNUNET_RAW}/{configs[ConfigKeys.DATASET_MASS.value]}/imagesTr",
    #     mass_dataset_labels=f"{NNUNET_RAW}/{configs[ConfigKeys.DATASET_MASS.value]}/labelsTr",
    #     tumour_dataset_labels=f"{NNUNET_RAW}/{configs[ConfigKeys.DATASET_TUMOUR.value]}/labelsTr",
    #     config=configs,
    #     observer=observer
    # )

    event_handler = KidneyPredictionHandler(
        folder=prediction_path,
        mass_dataset_images=f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/mass_inference",
        mass_dataset_labels=None,
        tumour_dataset_labels=None,
        config=configs,
        observer=observer,
        total_expected_cases = 1
    )

    observer.schedule(event_handler, prediction_path, recursive=False)
    observer.start()

    already_ready_predictions = glob.glob(f"{prediction_path}/*.nii.gz")
    for file in already_ready_predictions:
        event_handler.new_file_created(file)

    observer.join()
    csv_split = event_handler.crop_management.slice_management_df
    csv_split.to_csv(f"{ROOT_DIR}/{configs[ConfigKeys.KIDNEY_CHUNK_CSV.value]}")
