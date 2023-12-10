import os
from functools import partial

import tensorflow as tf
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

# # augmentations
# transforms = Compose([
#     Rotate(limit=40),
#     RandomBrightness(limit=0.1),
#     JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
#     HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
#     RandomContrast(limit=0.2, p=0.5),
#     HorizontalFlip(),
# ])
#
#
# def aug_fn(image):
#     data = {"image": image}
#     aug_data = transforms(**data)
#     aug_img = aug_data["image"]
#     aug_img = tf.cast(aug_img / 255.0, tf.float32)
#     # aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
#     return aug_img


datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2)

files_and_directories = os.listdir("../classification_module/datasets/dataset/test_set")
for ff in files_and_directories:
    for f in os.listdir("dataset/test_set/" + ff):
        img = load_img("dataset/test_set/" + ff + "/" + f)

        # x = tf.numpy_function(func=aug_fn, inp=[img], Tout=tf.float32)
        x = img_to_array(img)
        # Reshape the input image
        x = x.reshape((1,) + x.shape)
        i = 0

        # generate 15 new augmented images
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir="dataset/training_set/" + ff,
                                  save_prefix=ff,
                                  save_format='jpg'):
            i += 1
            if i > 15:
                break
