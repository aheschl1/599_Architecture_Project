import json
import os
from enum import Enum

import numpy as np
from scipy.ndimage import label, sum

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))

NNUNET_RAW = os.environ['nnUNet_raw']
NNUNET_RESULTS = os.environ['nnUNet_results']

SELF_CASE = "self case"
PARENT_CASE = "parent case"
PATH = "path"
TRANSFORMATION_PICKLE_PATH = "pkl"


def get_case_name_from_number(case: int) -> str:
    return '0' * (5 - len(str(case))) + str(case)


def get_configurations():
    with open(f"{ROOT_DIR}/config.json") as file:
        return json.load(file)


def assert_case_compatible(*args):
    old_case = "-1"
    for path in args:
        case = path.split('/')[-1].split('.')[0].split('_')[1]
        assert old_case == '-1' or case == old_case, f"Incompatible cases {case} and {old_case}."


def remove_small_connected(arr: np.array) -> np.array:
    """
    Given a numpy array, returns a new version which has had all unique elements below
    one quarter the total mass removed.
    """
    labeled_array, num_features = label(arr)
    volumes = np.zeros(num_features + 1)
    for i in range(1, num_features + 1):
        volume = sum(arr, labels=labeled_array, index=i)
        volumes[i] = volume

    threshold = np.sum(arr) // 4 # 20000
    mask_under_threshold = volumes < threshold
    labeled_array[mask_under_threshold[labeled_array]] = 0
    labeled_array[labeled_array != 0] = 1
    return labeled_array


class ConfigKeys(Enum):
    DATASET_TUMOUR = "dataset_tumour"
    DATASET_KIDNEY = "dataset_kidney"
    DATASET_MASS = "dataset_mass"
    KIDNEY_RESULTS = "kidney_results"
    MASS_RESULTS = "mass_results"
    TUMOUR_RESULTS = "tumour_results"
    KIDNEY_DILATION = "kidney_roi_dilation"
    MASS_DILATION = "mass_dilation"
    COMBINED_OUTPUT = "combined_output"
    KIDNEY_CHUNK_CSV = "kidney_chunk_csv"
    LABEL_LOOKUP = "label_lookup"
    RECHUNKED_MASS = "rechunked_mass"
    RECHUNKED_TUMOUR = "rechunked_tumour"
    INFERENCE = "inference"
