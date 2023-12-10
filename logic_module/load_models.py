import tensorflow as tf
from tensorflow import keras

planes_model = keras.models.load_model('./classification_module/output/model/model_planes')
planes_class_names = tf.keras.utils.image_dataset_from_directory('./classification_module/datasets/planes').class_names

tanks_model = keras.models.load_model('./classification_module/output/model/model_tanks')
tanks_class_names = tf.keras.utils.image_dataset_from_directory('./classification_module/datasets/tanks').class_names

fake_tanks_model = keras.models.load_model('./classification_module/output/model/model_fake_tanks')
fake_tanks_class_names = tf.keras.utils.image_dataset_from_directory('./classification_module/datasets/fake_tanks').class_names
