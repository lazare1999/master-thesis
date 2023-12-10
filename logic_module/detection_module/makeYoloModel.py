# !nvidia-smi
# !pip install ultralytics==8.0.28

"""install yolov8"""
from ultralytics import YOLO
from IPython.display import display, Image
import os
import ultralytics

from IPython import display

display.clear_output()
ultralytics.checks()
HOME = os.getcwd()
print(HOME)

"""Inference Example with Pretrained YOLOv8 Model"""
model = YOLO("yolov8x.pt")
model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', show=True, conf=0.25, save=True, task='detect')
# !yolo task=detect mode=predict model=yolov8x.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=True

Image(filename='runs/detect/predict/dog.jpeg', height=600)


"""Train YOLOv8 Model on Custom Dataset"""
# !mkdir {HOME}/datasets
# %cd {HOME}/datasets
#
# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="cBwIVBF4MUWe5FqlyCkw")
project = rf.workspace("university-of-georgia").project("military_tanks_planes_helicopters_ifv_apc_artilery_trucks")
dataset = project.version(1).download("yolov8")

model.train(data=f"{dataset.location}/data.yaml", epochs=25, imgsz=640, plots=True, task='detect')
# !yolo task=detect mode=train model=yolov8x.pt data={dataset.location}/data.yaml epochs=25 imgsz=640 plots=True

Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)

Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)

"""Validate Custom Model"""
customModel = YOLO("yolov8x.pt")
customModel.val(data=f"{dataset.location}/data.yaml", task='detect')
# !yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

"""Inference with Custom Model"""
customModel.predict(task='detect', conf=0.25, source=f"{dataset.location}/test/images", save=True)
# !yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True


import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict3/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")