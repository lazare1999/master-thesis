import numpy as np
import tensorflow as tf
from tensorflow.python.ops.confusion_matrix import confusion_matrix

image_size = (180, 180)

def predict_binary_class(img_array, model, class_names):
    img_array = tf.concat(img_array, axis=0)
    img_array = tf.image.resize(img_array, [180, 180])
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    # print(prediction)
    predictions = tf.where(prediction < 0.5, 0, 1)
    predicted_label = class_names[predictions[0].numpy()[0]]
    # print(predicted_label)

    if predicted_label =='real':
        return True

    return False
    #
    # result = confusion_matrix(predictions[0].numpy(), prediction)
    # print(result)

def predict_object(img_array, model, class_names):
    img_array = tf.concat(img_array, axis=0)
    img_array = tf.image.resize(img_array, [180, 180])
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)

    sorting = (-prediction).argsort()

    sorted_ = sorting[0][:1][0]

    predicted_label = class_names[sorted_]

    prob = (prediction[0][sorted_]) * 100
    prob = "%.2f" % round(prob, 2)
    # print("I have %s%% sure that it belongs to %s." % (prob, predicted_label))

    ans = "/ %s%% - %s." % (prob, predicted_label)
    return ans