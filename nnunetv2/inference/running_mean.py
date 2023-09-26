import torch
import numpy as np
import os
import SimpleITK as sitk

class PredictionManager:
    def __init__(self, output_dir:str, label_manager) -> None:
        self.output_dir = output_dir
        self.running_sums = None
        self.sample_count = 0
        self.label_manager = label_manager
    

    def __call__(self, probabilities_tensor:torch.Tensor)->None:
        if self.running_sums is None:
            self.running_sums = np.zeros_like(probabilities_tensor)
        self.running_sums += np.array(probabilities_tensor.cpu())
        self.sample_count += 1
    
    def done(self, case:str)->None:
        
        mean_prob = self.running_sums/self.sample_count
        segmentation = self.label_manager.convert_probabilities_to_segmentation(mean_prob)
        segmentation = sitk.GetImageFromArray(segmentation)

        sample_image = sitk.ReadImage(f"{self.output_dir}/case_{case}.nii.gz")
        segmentation.CopyInformation(sample_image)          # Add the code to get the image path, read the image using sitk ()
        
        new_case_path = os.path.join(self.output_dir, f"predictioncase_{case}")  # Entropies_folder is the path to the folder to store entropies map
        os.makedirs(new_case_path, exist_ok=True)
        sitk.WriteImage(segmentation, f"{new_case_path}/case_{case}.nii.gz")
