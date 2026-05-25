# backend/detection/detector.py
import cv2
from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize YOLO model for person detection.
        Args:
            model_name: YOLO model file (yolov8n.pt = nano, fastest)
            conf_threshold: Minimum confidence to consider a detection
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}
        
    def detect_people(self, frame):
        """
        Detect people in a single frame.
        Returns:
            list of bounding boxes: each is [x1, y1, x2, y2, confidence]
            also returns annotated frame with boxes drawn.
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        people_boxes = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class id
                    cls_id = int(box.cls[0])
                    # Check if it's a person (class 0 in COCO)
                    if cls_id == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        people_boxes.append([x1, y1, x2, y2, conf])
                        
                        # Draw bounding box and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"person {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return people_boxes, annotated_frame