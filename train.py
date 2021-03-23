from glob import glob
from utils import *
from config import *
from matplotlib import pyplot as plt
from model import unet
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train() -> None:
    """
    Train model with hyperparameters defined in config.py.
    """
    training_generator = MyGenerator(BATCH_SIZE, IMG_SIZE, train_images_paths, train_labels_paths)
    val_generator = MyGenerator(BATCH_SIZE, IMG_SIZE, val_images_paths, val_labels_paths)
    model = unet(IMG_SIZE, DROPOUT)
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=WBCE_LOSS, metrics=METRICS)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='loss', save_best_only=True)
    drop_alpha = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/' + '.'.join(model_name.split('.')[:-1]))
    model.fit(training_generator,
              validation_data=val_generator,
              epochs=EPOCHS,
              callbacks=[model_checkpoint, drop_alpha, tensorboard_callback])


def predict(paths: list, model_name: str) -> None:
    """
    Predict and save the results.

    @param paths: paths of images to predict
    @param model_name: saved hdf5 model filename
    """
    model = load_model(model_name, compile=False)
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=WBCE_LOSS, metrics=METRICS)
    for path in tqdm(paths):
        img = Image.open(path)
        input_arr = np.array(img.resize(IMG_SIZE, resample=Image.NEAREST))
        input_arr = normalize(input_arr)
        input_arr = input_arr.reshape([1, input_arr.shape[0], input_arr.shape[1], 1])
        prediction = model.predict(input_arr, batch_size=1)[0, :, :, 0]

        img = Image.fromarray(prediction)
        file_name = path.split(os.sep)[-1]
        img.save(os.path.join(RESULT_FOLDER, file_name))


if __name__ == '__main__':
    model_name = MODEL_NAME + datetime.now().strftime("%d.%m.%Y_%H.%M.%S") + '.hdf5'

    train_images_paths = glob(os.path.join(TR_IMAGES, '*.tiff'))
    train_labels_paths = glob(os.path.join(TR_LABELS, '*.tiff'))

    val_images_paths = glob(os.path.join(V_IMAGES, '*.tiff'))
    val_labels_paths = glob(os.path.join(V_LABELS, '*.tiff'))

    test_paths = glob(os.path.join(T_IMAGES, '*.tiff'))
    all_images_paths = glob(os.path.join(ALL_IMAGES, '*.tiff'))

    assert len(train_images_paths) == len(train_images_paths)
    assert len(val_images_paths) == len(val_labels_paths)

    train()
    predict(test_paths, model_name)
    predict(all_images_paths, model_name)
