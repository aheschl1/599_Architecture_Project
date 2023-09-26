import os
import numpy as np
import torch
import SimpleITK as sitk
import time
import gc

class EntropyManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        print(self.output_dir)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.entropy_sums = None

    def calculate_probabilities(self, prediction_array):
        # Ensure prediction_array is a 4D numpy array
        assert len(prediction_array.shape) == 4, "Input must be a 4D numpy array"

        # Initialize list of class tensors
        class_tensors = []

        # Iterate over classes
        for i in range(prediction_array.shape[0]):
            # Convert to tensor and clamp values
            class_tensor = prediction_array[i]
            class_tensor.clamp_(min=1e-9, max=1)
            class_tensor.mul_(-torch.log2(class_tensor))
            
            # Save the tensor for the class
            class_tensors.append(class_tensor)
            del class_tensor
            gc.collect()

        return class_tensors

    def __call__(self, probabilities_tensors):
        print(f"Shape of probabilities_tensor before calculate_probabilities: {probabilities_tensors.shape}")
        # Calculate the probabilities for each voxel in the prediction
        probabilities_tensors = self.calculate_probabilities(probabilities_tensors)
                
        #Do stuff with it
        if self.entropy_sums is None:
            # If this is the first prediction, use its shape and number of classes to initialize the entropy_sums list
            self.entropy_sums = [torch.zeros(probabilities_tensors[i].shape, device=probabilities_tensors[i].device) for i in range(len(probabilities_tensors))]

        for i, class_entropy in enumerate(probabilities_tensors):
            self.entropy_sums[i].add_(class_entropy)
    
    def done(self, case:str):
        masks = []
        for c in range(len(self.entropy_sums)):
            entropy_map = self.entropy_sums[c].cpu().numpy() # Convert to numpy
            entropy_image = sitk.GetImageFromArray(entropy_map)

            sample_image = sitk.ReadImage(f"{self.output_dir}/case_{case}.nii.gz")
            entropy_image.CopyInformation(sample_image)          # Add the code to get the image path, read the image using sitk ()
            
            masks.append(entropy_image)

        print("case index: ", case)
        new_case_path = os.path.join(self.output_dir, f"entropycase{case}")  # Entropies_folder is the path to the folder to store entropies map
        os.makedirs(new_case_path, exist_ok=True)

        for channel, mask in enumerate(masks):
            print(type(mask))
            mask_path = os.path.join(new_case_path, f"case{case}_channel{channel}.nii.gz")
            print(mask_path)
            sitk.WriteImage(mask, mask_path)
