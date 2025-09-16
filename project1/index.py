import cv2 as cv
from ultralytics import YOLO
import supervision as sv
import json

# Thresholds
conf = 0.65
iou = 0.8

model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator(thickness=8)
capture = cv.VideoCapture('video1_p1.mp4')
label_annotator = sv.LabelAnnotator(text_thickness=2)

num_detections = 0

name = "test" + '.mp4'
fourcc = cv.VideoWriter_fourcc(*'XVID') 
out = cv.VideoWriter('output.avi', fourcc, 20.0, (3840,2160))

while(True):
    # Running the loop to capture video
    isTrue , frame = capture.read()
    if not isTrue or frame is None:
        break

    # Capturing the results
    results = model.track(frame, tracker="bytetrack.yaml",conf = conf , iou = iou)
    
    # Extracting the detections
    detections = sv.Detections.from_ultralytics(results[0])
    print(detections)

    labels = []
    if detections.tracker_id is not None:
        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id
            in zip(detections.data["class_name"], detections.tracker_id) 
        ]
    else:
        labels = [f"{class_name}" for class_name in detections.data["class_name"]]
        
    # Calculating number of detections
    num_detections = len(detections.tracker_id) if detections.tracker_id is not None else len(detections.data["class_name"])

    frame = cv.putText(frame,"detections : " + str(num_detections),(0,100),cv.FONT_HERSHEY_TRIPLEX,1.0,(255,255,0),2)

    # Annotations and labeling
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    labeled_frame = label_annotator.annotate(
        annotated_frame.copy(),
        detections=detections,
        labels=labels
    )



    cv.imshow("Video",labeled_frame)
    out.write(labeled_frame)

    if(cv.waitKey(20)& 0xFF == ord('d')):
        capture.release()
        out.release()
        cv.destroyAllWindows()
        break

capture.release() 
out.release()
cv.destroyAllWindows()