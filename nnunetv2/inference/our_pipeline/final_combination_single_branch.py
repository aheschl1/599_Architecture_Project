import torch.nn.functional as F
import glob 
import SimpleITK as sitk
from multiprocessing.pool import Pool
from utils import ConfigKeys, ROOT_DIR, assert_case_compatible, get_configurations
from multiprocessing.pool import ThreadPool
from chunk_management.image_chunk_management import ImageCropManagement
import pandas as pd
import numpy as np
from tqdm import tqdm

class FinalHeirarchalCombination:
    def __init__(self, config: dict) -> None:
        self.output = f"{ROOT_DIR}/{config[ConfigKeys.COMBINED_OUTPUT.value]}"
        
        self.kidney_path = f"{ROOT_DIR}/{config[ConfigKeys.INFERENCE.value]}/kidney_predictions"
        self.tumour_path = f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/rechunked_tumour"
        self.mass_path = f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/rechunked_mass"

    
    def combine(self, predictions):
        
        def unpack_images() -> tuple[sitk.Image, sitk.Image, sitk.Image]:
            kidney_image = sitk.ReadImage(kidney_path)
            mass_image = sitk.ReadImage(mass_path) if mass_path is not None else None
            tumour_image = sitk.ReadImage(tumour_path) if tumour_path is not None else None

            return kidney_image, mass_image, tumour_image
        
        def get_image_arrays() -> tuple[np.array, np.array, np.array]:
            kidney_array = sitk.GetArrayFromImage(kidney_image)
            mass_array = sitk.GetArrayFromImage(mass_image) if mass_image is not None else None
            tumour_array = sitk.GetArrayFromImage(tumour_image) if tumour_image is not None else None

            return kidney_array, mass_array, tumour_array
        
        def merge_predictions() -> None:
            if mass_array is not None:
                mass_array[mass_array != 0] = 3
            if tumour_array is not None:
                tumour_array[tumour_array != 0] = 2
            kidney_array[kidney_array != 0] = 1
            
            if mass_array is not None and tumour_array is not None:
                mass_array[tumour_array != 0] = tumour_array[tumour_array != 0]
            if mass_array is not None:
                kidney_array[mass_array != 0] = mass_array[mass_array != 0]

        kidney_path = predictions['kidney']
        tumour_path = predictions['tumour']
        mass_path = predictions['masses']

        assert kidney_path is not None, "Kidney path must not be None."
        # kidney_path, mass_path, tumour_path = paths

        case_name_with_extension = kidney_path.split('/')[-1]
        print(f"Starting with {case_name_with_extension}")
        assert_case_compatible(*[x for x in [kidney_path, mass_path, tumour_path] if x is not None])
        #Case is compatible!
        kidney_image, mass_image, tumour_image = unpack_images()
        kidney_array, mass_array, tumour_array = get_image_arrays()
        merge_predictions()
        #Now kidney array holds all
        final_image = sitk.GetImageFromArray(kidney_array)
        final_image.CopyInformation(kidney_image)

        sitk.WriteImage(final_image, f"{self.output}/{case_name_with_extension}")
        print(f"Done with {case_name_with_extension}")

    def start(self):
        kidneys = sorted(glob.glob(f"{self.kidney_path}/*.nii.gz"))
        tumours = sorted(glob.glob(f"{self.tumour_path}/*.nii.gz"))
        masses = sorted(glob.glob(f"{self.mass_path}/*.nii.gz"))

        # print(kidneys[0], tumours[0], masses[0])
        # assert len(kidneys) == len(tumours) == len(masses), "Different number of masses, tumours, and kidney predictions."

        tasks = []
        print("Grouping paths")
        for prediction in tqdm(kidneys):
            file_name = prediction.split('/')[-1]
            this_task = {
                "kidney": prediction,
                "tumour": None,
                "masses": None
            }
            for prediction_t in tumours:
                if file_name in prediction_t:
                    this_task['tumour'] = prediction_t
                    break
            
            for prediction_m in masses:
                if file_name in prediction_m:
                    this_task['masses'] = prediction_m
                    break

            tasks.append(this_task)
        with Pool(16) as pool:
            pool.map(self.combine, tasks)

if __name__ == "__main__":
    configs = get_configurations()

    image_csv = pd.read_csv(f"{ROOT_DIR}/{configs[ConfigKeys.KIDNEY_CHUNK_CSV.value]}")
    #First we need to recombine the mass and tumour results.

    do_chunking = True

    if do_chunking:
        print("Working on rechunking the mass predictions...")
        ImageCropManagement.undo_splitting(
            image_csv,
            f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/rechunked_mass",
            "nii.gz",
            f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/mass_predictions",
        )
        #Now tumours
        print("Working on rechunking the tumour predictions...")
        ImageCropManagement.undo_splitting(
            image_csv,
            f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/rechunked_tumour",
            "nii.gz",
            f"{ROOT_DIR}/{configs[ConfigKeys.INFERENCE.value]}/tumour_predictions",
        )

    combiner = FinalHeirarchalCombination(configs)
    combiner.start()