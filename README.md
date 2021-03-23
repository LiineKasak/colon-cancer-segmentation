# Colon cancer segmentation

## Environment

The project is set up in an Anaconda environment. Details about versioning can be found in [environment.yml](https://github.com/LiineKasak/colon-cancer-segmentation/blob/master/environment.yml).

## 1. Data setup

The location of the given dataset must be specified in [config.py](https://github.com/LiineKasak/colon-cancer-segmentation/blob/master/config.py).
The base folder where images are generated can also be customized.

After verifying these two parameters, the first step is to run [load_data.py](https://github.com/LiineKasak/colon-cancer-segmentation/blob/master/load_data.py). This will generate folders with 32 bit tiff images divided into training and validation images and labels, and test images.

## 2. Training and prediction
When running [train.py](https://github.com/LiineKasak/colon-cancer-segmentation/blob/master/train.py). the model is trained with the chosen hyperparameters from [config.py](https://github.com/LiineKasak/colon-cancer-segmentation/blob/master/config.py), and then predictions are generated for all images from the dataset.

## 3. Analysis
In [eval_segmentation.ipynb](https://github.com/LiineKasak/colon-cancer-segmentation/blob/master/eval_segmentation.ipynb) we compare the IoU value over validation and training dataset while changing the bitmask threshold.

## 4. Visualization
Running [visualize.ipynb](https://github.com/LiineKasak/colon-cancer-segmentation/blob/master/visualize.ipynb), we can easily see the input image, ground truth and prediction on one line.

For the test set, the ground truth is empty since it is unknown.
