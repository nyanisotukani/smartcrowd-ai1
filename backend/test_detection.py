# test_detection.py
import cv2
from backend.detection.detector import PersonDetector

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize detector
    detector = PersonDetector(conf_threshold=0.5)
    
    print("Person detection running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Detect people
        people_boxes, annotated_frame = detector.detect_people(frame)
        
        # Show count on the frame
        count = len(people_boxes)
        cv2.putText(annotated_frame, f"People in frame: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display
        cv2.imshow("SmartCrowd AI - Person Detection", annotated_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    