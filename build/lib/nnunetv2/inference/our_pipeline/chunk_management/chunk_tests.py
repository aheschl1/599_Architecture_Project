import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, sum, find_objects
import pandas as pd
from threading import Thread
import os


def get_case_name_from_number(case:int)->str:
    return '0'*(5-len(str(case))) + str(case)

def remove_small_connected(arr:np.array)->np.array:
    """
    Given a numpy array, returns a new version which has had all unique elements below one quarter the total mass removed.
    """
    labeled_array, num_features = label(arr)
    volumes = np.zeros(num_features + 1)
    for i in range(1, num_features + 1):
        volume = sum(arr, labels=labeled_array, index=i)
        volumes[i] = volume

    threshold = 80000#np.sum(arr)//4
    mask_under_threshold = volumes < threshold
    labeled_array[mask_under_threshold[labeled_array]] = 0
    labeled_array[labeled_array!=0]=1
    return labeled_array


class BBox:
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z) -> None:
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z
    
    def get_cropped(self, *images)->np.array:
        cropped_images = []
        for image in images:
            cropped_images.append(image[
                self.min_x:self.max_x + 1,
                self.min_y:self.max_y + 1,
                self.min_z:self.max_z + 1
            ])
        return tuple(cropped_images)

class ImageCropManagment:
    """
    Object to manage the slicing and reconstruction of an image.
    """
    def __init__(self, output_folder:str, name_scheme:str, spacing=None, direction=None, origin=None, start_id=0, label_lookup=None) -> None:
        self.column_names=[
            "Case",
            "Parent Case",
            "Path",
            "Original X Shape",
            "Original Y Shape",
            "Original Z Shape",
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "min_z",
            "max_z"
        ]
        self.output_folder = output_folder
        self.name_scheme = name_scheme
        self._slice_management_df = pd.DataFrame(columns=self.column_names)
        self.spacing = spacing
        self.direction = direction
        self.origin = origin
        self._current_id = start_id
        self.label_lookup = label_lookup
    
    def new_image(self, basis:np.array, image:np.array, label:np.array, parent_case:str)->None:
        bounding_boxes = self.get_bounding_boxes(basis)
        print(f"We found {len(bounding_boxes)} objects in {parent_case}.")
        for bbox in bounding_boxes:
            new_id = self.current_case            
            name = self.name_scheme.replace('*', new_id)
            series = self._crop_save_image(
                image=image,
                label=label,
                bbox=bbox,
                case=name,
                parent_case=parent_case,
                original_shape=(image.shape[0], image.shape[1], image.shape[2])
            )
            self._slice_management_df = pd.concat([self.slice_management_df, series.to_frame().T], ignore_index=True)
        
    def _crop_save_image(self, image:np.array, label:np.array, case:str, bbox:BBox, parent_case:str, original_shape:tuple)->pd.Series:

        label_region = bbox.get_cropped(label)
        cropped_region = bbox.get_cropped(image)
        """
        Name, Parent ID, Save Path, minx, maxx, miny, maxy, minz, maxz
        """
        series = pd.Series([
            case,
            parent_case,
            f"{self.output_folder}/{case}",
            original_shape[0],
            original_shape[1],
            original_shape[2],
            bbox.min_x,
            bbox.max_x,
            bbox.min_y,
            bbox.max_y,
            bbox.min_z,
            bbox.max_z
        ],
        index=self.column_names
        )
        image = sitk.GetImageFromArray(cropped_region)
        if self.spacing != None:
            image.SetSpacing(self.spacing)
        if self.direction != None:
            image.SetDirection(self.direction)
        if self.origin != None:
            image.SetOrigin(self.origin)
        
        label = sitk.GetImageFromArray(label_region)
        if self.spacing != None:
            label.SetSpacing(self.spacing)
        if self.direction != None:
            label.SetDirection(self.direction)
        if self.origin != None:
            label.SetOrigin(self.origin)

        thread = Thread(
            target=ImageCropManagment.__write_image__,
            args=(image, f"{self.output_folder.replace('imagesTr', 'labelsTr')}/{case}",)
        )
        thread.start()

        case.replace('.', '_0000.', 1)
        thread = Thread(
            target=ImageCropManagment.__write_image__,
            args=(image, f"{self.output_folder}/{case}",)
        )
        thread.start()

        return series

    def get_bounding_boxes(self, image:np.array):
        """
        Returns a list of BBox objects which define where each object is located. NO MARGIN
        """
        labeled_array, _ = label(image)
        bounding_boxes = find_objects(labeled_array)
        boxes = []
        for i, box in enumerate(bounding_boxes):
            min_x = box[0].start
            max_x = box[0].stop - 1
            min_y = box[1].start
            max_y = box[1].stop - 1
            min_z = box[2].start
            max_z = box[2].stop - 1
            boxes.append(BBox(min_x, max_x, min_y, max_y, min_z, max_z))

        return boxes
    
    @staticmethod
    def undo_splitting(dataframe:pd.DataFrame, output_path:str, extension:str):
        samples_dataframes = dict(tuple(dataframe.groupby('ID')))
        for parent_case in samples_dataframes.values():
            ImageCropManagment.__attach__(parent_case, output_path, extension)
    
    @staticmethod
    def __attach__(samples:pd.DataFrame, output_path:str, extension:str)->None:
        reference = samples.iloc[0]
        container = np.zeros(shape=(reference['Original X Shape'], reference['Original Y Shape'], reference['Original Z Shape']))
        spacing, direction, origin = None, None, None
        for i, (_, row) in enumerate(samples.iterrows()):
            current_image = sitk.ReadImage(row['Path'])
            if i == 0:
                spacing=current_image.GetSpacing()
                direction=current_image.GetDirection()
                origin=current_image.GetOrigin()

            array = sitk.GetArrayFromImage(current_image)
            container[
                row['min_x']:row['max_x']+1,
                row['min_y']:row['max_y']+1,
                row['min_z']:row['max_z']+1
            ] = array

        container = sitk.GetImageFromArray(container)
        container.SetDirection(direction)
        container.SetSpacing(spacing)
        container.SetOrigin(origin)

        t = Thread(
            target=ImageCropManagment.__write_image__,
            args=(container,f"{output_path}/case_{get_case_name_from_number(reference['Parent Case'])}.{extension}",)
        )
        t.start()

    @staticmethod
    def __write_image__(image:sitk.Image, path:str)->None:
        """
        For writting an image from the background.
        """
        sitk.WriteImage(image, path)

    @property
    def slice_management_df(self):
        return self._slice_management_df
    @property
    def current_case(self):
        self._current_id += 1
        return get_case_name_from_number(self._current_id-1)
    
def main():
    target = "/home/andrewheschl/Documents/Temp/imagesTr_pred/Dataset029_Tumour"
    try:
        os.makedirs(f"{target}/imagesTr")
    except:
        pass

    try:
        os.makedirs(f"{target}/labelsTr")
    except:
        pass

    image = sitk.ReadImage("/home/andrewheschl/Documents/Temp/imagesTr_pred/case_00002.nii.gz")
    image_array = remove_small_connected(sitk.GetArrayFromImage(image))
    image_array[image_array!=0] = 1
    image_array = image_array
    
    management = ImageCropManagment(
        f"{target}/imagesTr", 
        name_scheme="case_*.nii.gz",
        spacing=image.GetSpacing(),
        direction=image.GetDirection(),
        origin=image.GetOrigin(),
        start_id = 0
    )

    management.new_image(image_array,"00002")

    df = management.slice_management_df
    df.to_csv('images.csv', index=False)

def undo():
    df = pd.read_csv('images.csv')
    ImageCropManagment.undo_splitting(df, "/home/andrewheschl/Documents/Temp/imagesTr_pred/Dataset029_Tumour/undone", extension="nii.gz")


if __name__ == "__main__":
    #main()
    undo()
            