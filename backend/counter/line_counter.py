# backend/counter/line_counter.py
import cv2
import numpy as np
from collections import defaultdict, deque

class LineCounter:
    def __init__(self, line_y_position=300, direction_check_frames=5):
        """
        Virtual line counter for entry/exit detection.
        Args:
            line_y_position: Y-coordinate of the horizontal line (in pixels)
            direction_check_frames: Number of frames to track before deciding direction
        """
        self.line_y = line_y_position
        self.direction_check_frames = direction_check_frames
        
        # Store last N positions per tracked ID
        self.tracked_positions = defaultdict(lambda: deque(maxlen=direction_check_frames))
        
        # Set of IDs that have already been counted for current crossing
        self.counted_crossings = set()
        
        # Counters
        self.total_entries = 0
        self.total_exits = 0
        self.current_occupancy = 0
        
    def update(self, tracked_objects, frame_shape):
        """
        Update counters based on current tracked objects.
        Args:
            tracked_objects: list of dicts with 'id' and 'bbox' [x1,y1,x2,y2]
            frame_shape: (height, width) of frame for line validation
        Returns:
            dict with updated counters (entries, exits, occupancy)
        """
        height, width = frame_shape[:2]
        # Ensure line is within frame
        line_y = min(max(self.line_y, 0), height)
        
        # Store current positions (center y-coordinate of each person)
        current_positions = {}
        for obj in tracked_objects:
            person_id = obj['id']
            bbox = obj['bbox']
            # Calculate center y (bottom of bounding box gives more stable crossing)
            center_y = (bbox[1] + bbox[3]) // 2  # (y1 + y2)/2
            current_positions[person_id] = center_y
            
            # Store position history
            self.tracked_positions[person_id].append(center_y)
        
        # Check for crossing events
        to_remove = []
        for person_id, positions in self.tracked_positions.items():
            if len(positions) < 2:
                continue  # Need at least 2 frames to detect direction
            
            # If already counted for current crossing, skip
            if person_id in self.counted_crossings:
                # Remove from counted after they've moved away from line
                if abs(positions[-1] - line_y) > 30:
                    to_remove.append(person_id)
                continue
            
            # Check crossing direction
            prev_y = positions[-2]
            curr_y = positions[-1]
            
            # Crossing from above -> below (downward) = Entry (assuming line is at entrance)
            # Crossing from below -> above (upward) = Exit
            crossed = False
            direction = None
            
            if prev_y <= line_y and curr_y > line_y:
                crossed = True
                direction = "down"   # Entry
            elif prev_y >= line_y and curr_y < line_y:
                crossed = True
                direction = "up"     # Exit
            
            if crossed:
                if direction == "down":
                    self.total_entries += 1
                    self.current_occupancy += 1
                elif direction == "up":
                    self.total_exits += 1
                    self.current_occupancy -= 1
                
                # Mark this ID as counted for this crossing
                self.counted_crossings.add(person_id)
                print(f"ID {person_id} crossed {'DOWN (Entry)' if direction=='down' else 'UP (Exit)'} | "
                      f"Entries: {self.total_entries}, Exits: {self.total_exits}, Occupancy: {self.current_occupancy}")
        
        # Clean up old counted IDs after they've moved away
        for pid in to_remove:
            self.counted_crossings.discard(pid)
        
        # Clean up positions for IDs no longer tracked
        current_ids = set(obj['id'] for obj in tracked_objects)
        for pid in list(self.tracked_positions.keys()):
            if pid not in current_ids:
                del self.tracked_positions[pid]
                self.counted_crossings.discard(pid)
        
        return {
            'entries': self.total_entries,
            'exits': self.total_exits,
            'occupancy': self.current_occupancy
        }
    
    def draw_line(self, frame):
        """Draw the virtual line on the frame."""
        height, width = frame.shape[:2]
        line_y = min(max(self.line_y, 0), height)
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
        cv2.putText(frame, "ENTRY/EXIT LINE", (10, line_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return frame
    
    def draw_stats(self, frame):
        """Draw entry/exit/occupancy stats on frame."""
        cv2.putText(frame, f"Entries: {self.total_entries}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {self.total_exits}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Inside: {self.current_occupancy}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame