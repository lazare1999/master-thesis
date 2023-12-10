from logic_module.classification_module.recognition import predict_binary_class, predict_object
from logic_module.detection_module.detection import parse_arguments, box_annotator
import cv2

import tensorflow as tf
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import pafy

import load_models



def start_app(url):
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    if url is not None:
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")

        cap = cv2.VideoCapture(best.url)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("detection_module/best.pt")

    ans =""

    while True:
        ret, f = cap.read()

        result = model(source=f, agnostic_nms=True, conf=0.75)[0]

        boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
        confidences = result.boxes.conf.to('cpu').numpy().astype(float)
        labels = result.boxes.cls.to('cpu').numpy().astype(int)

        d = sv.Detections.from_yolov8(result)

        for box, conf, label in zip(boxes, confidences, labels):
            x_min, y_min, x_max, y_max = box
            image_crop = f[y_min:y_max, x_min:x_max]

            # {0: 'tank_above', 1: 'tank_back', 2: 'tank_front', 3: 'tank_side'}
            if label in [0, 1, 2, 3] and predict_binary_class(image_crop,
                                                              load_models.fake_tanks_model,
                                                              load_models.fake_tanks_class_names):

                ans = predict_object(image_crop, load_models.tanks_model, load_models.tanks_class_names)


        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f} {ans}"
            for _, confidence, class_id, _
            in d
        ]
        f = box_annotator.annotate(
            scene=f,
            detections=d,
            labels=labels
        )

        if ret:
            cv2.imshow("cam1", f)

        if cv2.waitKey(30) == 27:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q To exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # დროებითია
    start_app("https://www.youtube.com/watch?v=iO4Y8cAKcUw&t=40s")





