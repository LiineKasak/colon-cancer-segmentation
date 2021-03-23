import os
from utils import weighted_bce, f1, IoU

# TO CUSTOMIZE
# project location
BASE_FOLDER = 'C:\\Users\\liine\\Documents\\ML4H\\P1\\'
# downloaded dataset in the project folder
SOURCE_FOLDER = f"{BASE_FOLDER}{os.sep}ml4h_proj1_colon_cancer_ct"

# MODEL DETAILS, HYPER-PARAMETERS
MODEL_NAME = 'unet'
IMG_SIZE = (256, 256)
BATCH_SIZE = 12
EPOCHS = 40
LEARNING_RATE = 1e-3
DROPOUT = 9 * [0.25]
WEIGHT_ZEROS = 0.11
WEIGHT_ONES = 1
WBCE_LOSS = lambda y_true, y_pred: weighted_bce(y_true, y_pred, weight0=WEIGHT_ZEROS, weight1=WEIGHT_ONES)
METRICS = [f1, IoU]

# NON-CHANGEABLE
DATA_FOLDER = f"{BASE_FOLDER}{os.sep}images"
RESULT_FOLDER = f"{BASE_FOLDER}{os.sep}results"
ALL_LABELS = f"{DATA_FOLDER}{os.sep}training_labels"
ALL_IMAGES = f"{DATA_FOLDER}{os.sep}training_images"
TR_LABELS = f"{DATA_FOLDER}{os.sep}training_labels_cancer"
TR_IMAGES = f"{DATA_FOLDER}{os.sep}training_images_cancer"
V_LABELS = f"{DATA_FOLDER}{os.sep}validation_labels_cancer"
V_IMAGES = f"{DATA_FOLDER}{os.sep}validation_images_cancer"
T_IMAGES = f"{DATA_FOLDER}{os.sep}test_images"
