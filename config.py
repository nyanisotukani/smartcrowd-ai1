# config.py
import os

# Camera source (0 for built-in/webcam, or IP camera URL)
CAMERA_SOURCE = 0

# Detection confidence threshold (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Entry/exit line position (y-coordinate in pixels)
LINE_POSITION = 300

# Database file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "smartcrowd.db")

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000