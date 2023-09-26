import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, sum, find_objects
import pandas as pd
from threading import Thread
import os
from utils import get_case_name_from_number
from tqdm import tqdm

# TODO remove this, it will be performed in other steps
def remove_small_connected(arr: np.array) -> np.array:
    """
    Given a numpy array, returns a new version which has had all unique elements below one quarter the total mass removed.
    """
    labeled_array, num_features = label(arr)
    volumes = np.zeros(num_features + 1)
    for i in range(1, num_features + 1):
        volume = sum(arr, labels=labeled_array, index=i)
        volumes[i] = volume

    threshold = 80000  # np.sum(arr)//4
    mask_under_threshold = volumes < threshold
    labeled_array[mask_under_threshold[labeled_array]] = 0
    labeled_array[labeled_array != 0] = 1
    return labeled_array


class BBox:
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z) -> None:
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z

    def get_cropped(self, *images) -> np.array:
        cropped_images = []
        for image in images:
            cropped_images.append(
                image[
                self.min_x: self.max_x + 1,
                self.min_y: self.max_y + 1,
                self.min_z: self.max_z + 1,
                ]
            )
        return tuple(cropped_images)


class ImageCropManagement:
    """
    Object to manage the slicing and reconstruction of an image.
    """

    def __init__(
            self,
            output_folder: str,
            label_output_folder: list[str],
            name_scheme: str,
            name_scheme_label: str,
            spacing=None,
            direction=None,
            origin=None,
            start_id=0,
            extension="nii.gz",
    ) -> None:
        self.column_names = [
            "Case",
            "Label Case",
            "Parent Case",
            "Path",
            "Label Path",
            "Original X Shape",
            "Original Y Shape",
            "Original Z Shape",
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "min_z",
            "max_z",
        ]
        self.output_folder = output_folder
        self.name_scheme = name_scheme
        self.name_scheme_label = name_scheme_label
        self._slice_management_df = pd.DataFrame(columns=self.column_names)
        self.spacing = spacing
        self.direction = direction
        self.origin = origin
        self._current_id = start_id
        self.extension = extension
        self.label_output_folder = label_output_folder

    def new_image(
            self, basis: np.array, image: np.array, labels: list[np.array], parent_case: str
    ) -> None:
        # Bounding boxes around the unique regions
        bounding_boxes = self.get_bounding_boxes(basis)
        del basis
        print(f"We found {len(bounding_boxes)} objects in {parent_case}.")
        for bbox in bounding_boxes:
            new_id = self.current_case
            name_image = self.name_scheme.replace("*", new_id)
            name_label = self.name_scheme_label.replace("*", new_id)
            series = self._crop_save_image(
                image=image,
                labels=labels,
                bbox=bbox,
                name_image=name_image,
                name_label=name_label,
                parent_case=parent_case,
                original_shape=image.shape,
            )
            self._slice_management_df = pd.concat(
                [self.slice_management_df, series.to_frame().T],
                ignore_index=True
            )

    def _crop_save_image(
            self,
            image: np.array,
            labels: list[np.array],
            name_image: str,
            name_label: str,
            bbox: BBox,
            parent_case: str,
            original_shape: tuple[int, int, int],
    ) -> pd.Series:
        cropped_region = bbox.get_cropped(image)[0]
        print(type(cropped_region))
        # print(label_region.shape, cropped_region.shape)
        """
        Name, Parent ID, Save Path, minx, maxx, miny, maxy, minz, maxz
        """
        series = pd.Series(
            [
                name_image,
                name_label,
                parent_case,
                f"{self.output_folder}/{name_image}.{self.extension}",
                f"{self.label_output_folder[0]}/{name_label}.{self.extension}",
                original_shape[0],
                original_shape[1],
                original_shape[2],
                bbox.min_x,
                bbox.max_x,
                bbox.min_y,
                bbox.max_y,
                bbox.min_z,
                bbox.max_z,
            ],
            index=self.column_names,
        )
        image = sitk.GetImageFromArray(cropped_region)
        if self.spacing is not None:
            image.SetSpacing(self.spacing)
        if self.direction is not None:
            image.SetDirection(self.direction)
        if self.origin is not None:
            image.SetOrigin(self.origin)

        Thread(
            target=ImageCropManagement._write_image,
            args=(
                image,
                f"{self.output_folder}/{name_image}.{self.extension}",
            ),
        ).start()

        for i, label in enumerate(labels):
            label = sitk.GetImageFromArray(bbox.get_cropped(label)[0])
            if self.spacing is not None:
                label.SetSpacing(self.spacing)
            if self.direction is not None:
                label.SetDirection(self.direction)
            if self.origin is not None:
                label.SetOrigin(self.origin)

            Thread(
                target=ImageCropManagement._write_image,
                args=(
                    label,
                    f"{self.label_output_folder[i]}/{name_label}.{self.extension}",
                ),
            ).start()

        return series

    @staticmethod
    def get_bounding_boxes(image: np.array) -> list[BBox]:
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
    def undo_splitting(
            dataframe: pd.DataFrame,
            output_path: str,
            extension: str,
            prediction_root: str,
            use_label_name=True,
    ):
        samples_dataframes = dict(tuple(dataframe.groupby("Parent Case")))
        for parent_case in tqdm(samples_dataframes.values()):
            ImageCropManagement._attach(
                parent_case, output_path, extension, prediction_root, use_label_name
            )

    @staticmethod
    def _attach(
            samples: pd.DataFrame,
            output_path: str,
            extension: str,
            prediction_root: str,
            use_label_name,
    ) -> None:
        reference = samples.iloc[0]
        container = np.zeros(
            shape=(
                reference["Original X Shape"],
                reference["Original Y Shape"],
                reference["Original Z Shape"],
            )
        )
        spacing, direction, origin = None, None, None
        for i, (_, row) in enumerate(samples.iterrows()):
            file_name = (
                f"{row['Case']}.{extension}"
                if not use_label_name
                else f"{row['Label Case']}.{extension}"
            )
            try:
                current_image = sitk.ReadImage(f"{prediction_root}/{file_name}")
            except RuntimeError:
                print(f"Could not find {file_name}. Failed to fully rebuild parent case_{row['Parent Case']}.")
                continue

            spacing = current_image.GetSpacing()
            direction = current_image.GetDirection()
            origin = current_image.GetOrigin()

            array = sitk.GetArrayFromImage(current_image)
            # print(file_name, array.shape, row['Parent Case'])

            try:
                container[
                    row["min_x"]: row["max_x"] + 1,
                    row["min_y"]: row["max_y"] + 1,
                    row["min_z"]: row["max_z"] + 1,
                ] = array
            except ValueError:
                print(f"Failure on {file_name} with parent {row['Parent Case']}")

        container = sitk.GetImageFromArray(container)
        if direction is not None:
            container.SetDirection(direction)
            container.SetSpacing(spacing)
            container.SetOrigin(origin)

        t = Thread(
            target=ImageCropManagement._write_image,
            args=(
                container,
                f"{output_path}/case_{get_case_name_from_number(reference['Parent Case'])}.{extension}",
            ),
        )
        t.start()

    @staticmethod
    def _write_image(image: sitk.Image, path: str) -> None:
        """
        For writting an image from the background.
        """
        print("Writting", path)
        sitk.WriteImage(image, path)

    @property
    def slice_management_df(self):
        return self._slice_management_df

    @property
    def current_case(self):
        self._current_id += 1
        return get_case_name_from_number(self._current_id - 1)


def main():
    target = "/home/andrewheschl/Documents/Temp/imagesTr_pred/Dataset029_Tumour"
    try:
        os.makedirs(f"{target}/imagesTr")
    except OSError:
        pass

    try:
        os.makedirs(f"{target}/labelsTr")
    except OSError:
        pass

    image = sitk.ReadImage(
        "/home/andrewheschl/Documents/Temp/imagesTr_pred/case_00002.nii.gz"
    )
    image_array = remove_small_connected(sitk.GetArrayFromImage(image))
    image_array[image_array != 0] = 1
    image_array = image_array

    management = ImageCropManagement(
        f"{target}/imagesTr",
        f"{target}/labelsTr",
        name_scheme="case_*_0000",
        name_scheme_label="case_*",
        spacing=image.GetSpacing(),
        direction=image.GetDirection(),
        origin=image.GetOrigin(),
        start_id=0,
    )
    import copy

    management.new_image(
        basis=image_array,
        parent_case="00002",
        image=copy.copy(image_array),
        label=copy.copy(image_array),
    )

    df = management.slice_management_df
    df.to_csv("images.csv", index=False)


def undo():
    df = pd.read_csv("images.csv")
    ImageCropManagement.undo_splitting(
        df,
        "/home/andrewheschl/Documents/Temp/imagesTr_pred/Dataset029_Tumour/undone",
        extension="nii.gz",
        prediction_root="/home/andrewheschl/Documents/Temp/imagesTr_pred/Dataset029_Tumour/predictions",
        use_label_name=True,
    )


if __name__ == "__main__":
    # main()
    undo()
