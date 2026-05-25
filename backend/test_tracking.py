# test_tracking.py
import cv2
from backend.detection.tracker import PersonTracker

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    tracker = PersonTracker(conf_threshold=0.5)
    print("Tracking running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracked_objects, annotated_frame = tracker.update(frame)
        
        # Show count of unique people currently tracked
        count = len(tracked_objects)
        cv2.putText(annotated_frame, f"People: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Optional: list IDs for debugging
        ids = [obj['id'] for obj in tracked_objects]
        cv2.putText(annotated_frame, f"IDs: {ids}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("SmartCrowd AI - Tracking", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()