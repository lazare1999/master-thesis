import os

import joblib
from keras import layers

from models import construct_model_1, construct_model_2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras

image_size = (180, 180)
batch_size = 5
epochs = 100

def save_model(dataset, name, constr_model):
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical",
                              input_shape=image_size + (3,)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(factor=0.2),
            layers.RandomContrast(factor=0.2)
        ]
    )

    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img_g, label: (data_augmentation(img_g), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    if constr_model ==1:
        model = construct_model_1(input_shape=image_size + (3,), num_classes=num_classes)
    else:
        model = construct_model_2(input_shape=image_size + (3,), num_classes=num_classes)

    callback_dir = f"output/callbacks/callbacks_{name}"
    callbacks = [
        keras.callbacks.ModelCheckpoint(callback_dir + "/save_at_{epoch}.keras"),
        keras.callbacks.TensorBoard(log_dir="output/callbacks/tensorboard/" + name)
    ]

    loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if num_classes == 2:
        loss =tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model_name = f"model_{name}"
    model.save('output/model/' + model_name)

    history_name = f"history_{name}"
    joblib.dump(history, 'output/history/' + history_name)

