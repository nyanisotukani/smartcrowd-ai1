# backend/detection/tracker.py
from ultralytics import YOLO
import cv2

class PersonTracker:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5, iou_threshold=0.5):
        """
        Initialize YOLO model with built-in tracking.
        Args:
            model_name: YOLO model file
            conf_threshold: Minimum confidence for detection
            iou_threshold: IOU threshold for tracking (ignored by ByteTrack but kept for consistency)
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
        # Store track history
        self.track_history = {}  # track_id -> list of (frame_num, bbox)
        self.frame_count = 0
        
    def update(self, frame):
        """
        Process a frame, return tracked people with IDs and bounding boxes.
        Args:
            frame: numpy array (BGR)
        Returns:
            tracked_objects: list of dicts [{'id': int, 'bbox': [x1,y1,x2,y2], 'conf': float}]
            annotated_frame: frame with boxes and IDs drawn
        """
        self.frame_count += 1
        
        # Run tracking (persist=True keeps track IDs across frames)
        # tracker='bytetrack.yaml' uses ByteTrack algorithm
        results = self.model.track(frame, 
                                   conf=self.conf_threshold, 
                                   persist=True, 
                                   tracker='bytetrack.yaml',
                                   verbose=False)
        
        tracked_objects = []
        annotated_frame = frame.copy()
        
        # Check if tracking succeeded and boxes exist
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                if cls == 0:  # person only
                    x1, y1, x2, y2 = map(int, box)
                    tracked_objects.append({
                        'id': int(track_id),
                        'bbox': [x1, y1, x2, y2],
                        'conf': float(conf)
                    })
                    
                    # Store in history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append((self.frame_count, [x1, y1, x2, y2]))
                    
                    # Draw bounding box with ID
                    color = (0, 255, 0)  # green
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return tracked_objects, annotated_frame
    
    def get_active_ids(self):
        """Return set of track IDs currently being tracked"""
        return set([obj['id'] for obj in self.tracked_objects]) if hasattr(self, 'tracked_objects') else set()