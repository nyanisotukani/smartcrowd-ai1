# test_counting.py
import cv2
from backend.detection.tracker import PersonTracker
from backend.counter.line_counter import LineCounter

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize tracker and line counter
    tracker = PersonTracker(conf_threshold=0.5)
    # Adjust line position based on your camera view. 300 is a starting point.
    line_counter = LineCounter(line_y_position=300, direction_check_frames=5)
    
    print("Line counting running. Press 'q' to quit.")
    print("Cross the line from top to bottom = Entry, bottom to top = Exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get tracked objects
        tracked_objects, annotated_frame = tracker.update(frame)
        
        # Update counters
        stats = line_counter.update(tracked_objects, frame.shape)
        
        # Draw line and stats
        annotated_frame = line_counter.draw_line(annotated_frame)
        annotated_frame = line_counter.draw_stats(annotated_frame)
        
        # Show additional info: current tracked IDs
        ids = [obj['id'] for obj in tracked_objects]
        cv2.putText(annotated_frame, f"Active IDs: {ids}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("SmartCrowd AI - Entry/Exit Counter", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal stats - Entries: {line_counter.total_entries}, Exits: {line_counter.total_exits}, Occupancy: {line_counter.current_occupancy}")

if __name__ == "__main__":
    main()