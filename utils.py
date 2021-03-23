import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence


def weighted_bce(y_true: np.ndarray, y_pred: np.ndarray, weight1: float = 1.0, weight0: float = 1.0):
    """
    Generate a loss function for binary cross-entropy with custom weights.

    @param y_true: ground truth
    @param y_pred: prediction
    @param weight1: weight for 1s
    @param weight0: weight for 0s
    @return: weighted binary cross-entropy loss function
    """
    weights = (1.0 - y_true) * weight0 + y_true * weight1
    bce = K.binary_crossentropy(y_true, y_pred)
    w_bce = K.mean(weights * bce)
    return w_bce


def IoU(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0):
    """
    Intersection over Union metric.

    Computes the intersection over union, a metric for

    @param y_true: ground truth
    @param y_pred: prediction
    @param smooth: small integer to avoid division by zero
    @return: calculated intersection over union
    """
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou


def f1(y_true: np.ndarray, y_pred: np.ndarray):
    """
    F1 score metric.

    Computes the F1 score, a weighted average of precision and recall.

    @param y_true: ground truth
    @param y_pred: prediction
    @return: calculated F1 score
    """

    def recall(y_true: np.ndarray, y_pred: np.ndarray):
        """ Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true: np.ndarray, y_pred: np.ndarray):
        """ Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def normalize(arr: np.ndarray):
    """
    Normalize array to have values between 0 and 1.
    @param arr: input array to normalize
    @param th: threshold above which values are converted to 1s
    @return: normalized array
    """
    arr -= np.min(arr)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max != 0:
        arr = arr / (arr_max - arr_min)
    return arr


class MyGenerator(Sequence):
    """
    Custom data generator, given directories of data and labels.
    """

    def __init__(self, batch_size: int, img_size: tuple, input_img_paths: list, labels_img_paths: list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = np.array(input_img_paths)
        self.labels_img_paths = np.array(labels_img_paths)
        self.indices = np.arange(self.input_img_paths.shape[0])

    def __len__(self) -> int:
        """
        @return: number of batches to process
        """
        return len(self.labels_img_paths) // self.batch_size

    def __getitem__(self, idx: int):
        """
        Get a new batch.

        @param idx: index of batch process
        @return: new batch corresponding to idx
        """
        i = idx * self.batch_size
        idxs = self.indices[i: i + self.batch_size]
        batch_input_img_paths = self.input_img_paths[idxs]
        batch_label_img_paths = self.labels_img_paths[idxs]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = Image.open(path)
            arr = np.array(img.resize(self.img_size, resample=Image.NEAREST))
            x[j] = normalize(arr)
        y = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_label_img_paths):
            img = Image.open(path)
            arr = np.array(img.resize(self.img_size, resample=Image.NEAREST))
            y[j] = normalize(arr, th=0.5)
        return x, y

    def on_epoch_end(self):
        """
        Shuffle data on epoch end.
        """
        np.random.shuffle(self.indices)
