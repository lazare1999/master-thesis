import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.utils import img_to_array
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.shell import sc, spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.functions import udf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# sparkSession = SparkSession.builder.appName("pyspark").getOrCreate()

sparkSession = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "30g") \
    .appName('my-app') \
    .getOrCreate()

# Read from HDFS
# df_load = sparkSession.read.format("image").option("dropInvalid", True).load("hdfs://localhost:9000/user/lazo/datasets/tanks/*")
# df_load.select("image.origin", "image.width", "image.height").show(truncate=False)
# df_load.show()

# read in the files from the mounted storage as binary file
df_load = sparkSession.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpeg") \
  .option("recursiveFileLookup", "true") \
  .load('hdfs://localhost:9000/user/lazo/datasets/tanks/*')

# select the base model, here I have used ResNet50
model = ResNet50(include_top=False)
model.summary()  # verify that the top layer is removed

bc_model_weights = sc.broadcast(model.get_weights())


# declaring functions to execute on the worker nodes of the Spark cluster
def model_fn():
    """
    Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
    """
    m = ResNet50(weights=None, include_top=False)
    m.set_weights(bc_model_weights.value)
    return m


def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize((224, 224))
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(m, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    i = np.stack(content_series.map(preprocess))
    preds = m.predict(i)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)


@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    """
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                                is a pandas Series of image data.
    """
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    m = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(m, content_series)



# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "3")

# We can now run featurization on our entire Spark DataFrame.
# NOTE: This can take a long time (about 10 minutes) since it applies a large model to the full dataset.
features_df = df_load.repartition(16).select(col("path"), featurize_udf("content").alias("features"))

# MLLib needs some post processing of the features column format
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
features_df = features_df.select(
    col("path"),
    list_to_vector_udf(features_df["features"]).alias("features")
)

# OMITTED HERE
# You need to add the labels to your dataset based on the path of your images

# splitting in to training, validate and test set
df_train_split, df_validate_split, df_test_split = features_df.randomSplit([0.6, 0.3, 0.1], 42)

# Here we start to train the tail of the model

# This concatenates all feature columns into a single feature vector in a new column "featuresModel".
vectorAssembler = VectorAssembler(inputCols=['features'], outputCol="featuresModel")

labelIndexer = StringIndexer(inputCol="path", outputCol="indexedTarget").fit(features_df)

lr = LogisticRegression(maxIter=5, regParam=0.03,
                        elasticNetParam=0.5, labelCol="indexedTarget", featuresCol="featuresModel")

# define a pipeline model
sparkdn = Pipeline(stages=[labelIndexer, vectorAssembler, lr])
spark_model = sparkdn.fit(df_train_split)  # start fitting or training

# evaluating the model
predictions = spark_model.transform(df_test_split)

# Select example rows to display.
predictions.select("prediction", "indexedTarget", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedTarget", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))


# evaluate the model with test set
evaluator = MulticlassClassificationEvaluator()
tx_test = spark_model.transform(df_test_split)
print('F1-Score ', evaluator.evaluate(
                          tx_test,
                          {evaluator.metricName: 'f1'})
)
print('Precision ', evaluator.evaluate(
                          tx_test,
                          {evaluator.metricName: 'weightedPrecision'})
)
print('Recall ', evaluator.evaluate(
                          tx_test,
                          {evaluator.metricName: 'weightedRecall'})
)
print('Accuracy ', evaluator.evaluate(
                          tx_test,
                          {evaluator.metricName: 'accuracy'})
)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)
      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.

      for i, j in itertools.product(
          range(cm.shape[0]), range(cm.shape[1])
      ):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')



y_true = tx_test.select("label")
y_true = y_true.toPandas()

y_pred = tx_test.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=range(10))

sns.set_style("darkgrid")
plt.figure(figsize=(7,7))
plt.grid(False)

# call pre defined function
plot_confusion_matrix(cnf_matrix, classes=range(10))

def multiclass_roc_auc_score(y_test, y_prd, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_prd = lb.transform(y_prd)
    return roc_auc_score(y_test, y_prd, average=average)

print('ROC AUC score:', multiclass_roc_auc_score(y_true,y_pred))

# all columns after transformations
print(tx_test.columns)

# see some predicted output
tx_test.select('image', "prediction", "label").show()