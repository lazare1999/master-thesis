import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args



def detection():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # cap2 = cv2.VideoCapture(2)
    # cap2.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8x.pt")



    # zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    # zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    # zone_annotator = sv.PolygonZoneAnnotator(
    #     zone=zone,
    #     color=sv.Color.red(),
    #     thickness=2,
    #     text_thickness=4,
    #     text_scale=2
    # )

    while True:
        ret, frame = cap.read()
        # ret2, frame2 = cap2.read()

        result = model(source=frame, agnostic_nms=True, conf=0.75)[0]
        detections = sv.Detections.from_yolov8(result)
        # detections = detections[(detections.class_id == 0)]

        boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
        confidences = result.boxes.conf.to('cpu').numpy().astype(float)
        labels = result.boxes.cls.to('cpu').numpy().astype(int)

        for box, conf, label in zip(boxes, confidences, labels):
            x_min, y_min, x_max, y_max = box
            image_crop = frame[y_min:y_max, x_min:x_max]
            # TODO : ეს გადაეცი კლასიფიკაციას
            im = Image.fromarray(image_crop)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # zone.trigger(detections=detections)
        # frame = zone_annotator.annotate(scene=frame)

        if ret:
            cv2.imshow("cam1", frame)

        # if ret2:
        #     cv2.imshow("cam2", frame2)

        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    detection()