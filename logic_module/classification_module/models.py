import os

from keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import tensorflow as tf
from tensorflow import keras


def construct_model_1(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def construct_model_2(input_shape, num_classes):
    s_model = tf.keras.Sequential()
    s_model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    s_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    s_model.add(MaxPool2D((2, 2)))
    s_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    s_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    s_model.add(MaxPool2D((2, 2)))
    s_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    s_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    s_model.add(MaxPool2D((2, 2)))
    s_model.add(layers.Dropout(0.4))
    s_model.add(Flatten())
    s_model.add(Dense(256, activation='relu'))
    s_model.add(Dense(256, activation='relu'))

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    s_model.add(Dense(units, activation=activation))
    return s_model