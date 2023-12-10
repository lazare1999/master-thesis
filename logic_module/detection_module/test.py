from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
from imageai.Detection import VideoObjectDetection


def analyze(source, model_location):
    model = YOLO(model_location)
    results = model.predict(source=source, show=True, save=True, save_crop=True, conf=0.6)
    print(results)