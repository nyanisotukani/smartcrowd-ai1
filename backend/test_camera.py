# test_camera.py
import cv2

print("Opening webcam...")
cap = cv2.VideoCapture(0)  # 0 = default USB/webcam

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera OK. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show the live feed
    cv2.imshow("SmartCrowd AI - Camera Test", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera test finished.")