from os.path import join
import numpy as np
import nibabel as nib
from PIL import Image
from config import *
from tqdm import tqdm
from glob import glob
from shutil import move, copyfile
from utils import normalize


def create_directories() -> None:
    """
    Create necessary directories for data.
    """
    for folder in [DATA_FOLDER, RESULT_FOLDER, ALL_IMAGES, ALL_LABELS,
                   TR_IMAGES, TR_LABELS, V_IMAGES, V_LABELS, T_IMAGES]:
        if not os.path.exists(folder):
            os.makedirs(folder)


def save_slices(np_volume: np.ndarray, save_path: str, filename: str) -> None:
    """
    Save slices of a 3D numpy array as tiff files.

    @param np_volume: 3D numpy array
    @param save_path: directory where to save slices
    @param filename
    """
    n = np_volume.shape[2]
    for i in range(n):
        slab = np_volume[:, :, i]
        img = Image.fromarray(slab)
        new_file_name = filename + '_slice_{}.tiff'.format(i)
        img.save(join(save_path, new_file_name))


def read_training_data() -> None:
    """
    Read training data from given dataset and save slices of all nii.gz files as tiff files.
    """
    for f in tqdm(glob(join(SOURCE_FOLDER, 'imagesTr', '*.nii.gz'))):
        filename = f.split(os.sep)[-1]
        img_array = nib.load(join(SOURCE_FOLDER, 'imagesTr', filename)).get_fdata()
        lbl_array = nib.load(join(SOURCE_FOLDER, 'labelsTr', filename)).get_fdata()
        file_name = f.split(os.sep)[-1].split('.')[0]
        save_slices(img_array, ALL_IMAGES, file_name)
        save_slices(lbl_array, ALL_LABELS, file_name)


def read_testing_data() -> None:
    """
    Read testing data from given dataset and save slices of all nii.gz files as tiff files.
    """
    for f in tqdm(glob(join(SOURCE_FOLDER, 'imagesTs', '*.nii.gz'))):
        img_array = nib.load(join(SOURCE_FOLDER, 'imagesTs', f)).get_fdata()
        file_name = f.split(os.sep)[-1].split('.')[0]
        save_slices(img_array, T_IMAGES, file_name)


def split_training_data(validation_split: float = 0.2) -> None:
    """
    Generate a split of training and validation data given a set of data and labels.
    Test and validation images and labels will be copied to a new folder given by config.
    """
    assert 0 < validation_split < 1
    labels = np.array(glob(join(ALL_LABELS, '*.tiff')))
    for path in tqdm(labels):
        img = Image.open(path)
        arr = normalize(np.array(img.resize(IMG_SIZE, resample=Image.NEAREST)))
        if np.count_nonzero(arr > 0) > 0:
            slice_name = path.split(os.sep)[-1]
            copyfile(path, join(TR_LABELS, slice_name))
            copyfile(join(ALL_IMAGES, slice_name), join(TR_IMAGES, slice_name))

    images = glob(join(TR_IMAGES, '*.tiff'))
    size = len(images)
    choice = np.random.choice(np.arange(size), int(validation_split * size), replace=False)
    chosen_images = np.array(images)[choice]

    for img in tqdm(chosen_images):
        slice_name = img.split(os.sep)[-1]
        move(img, join(V_IMAGES, slice_name))
        move(join(TR_LABELS, slice_name), join(V_LABELS, slice_name))


if __name__ == '__main__':
    #create_directories()
    #read_training_data()
    read_testing_data()
    #split_training_data()
