from tensorflow.keras.models import load_model
import tensorflow as tf


def model_load():
    model_1 = load_model('../Age_detection_19.h5')
    model_1.summary()
    return model_1





