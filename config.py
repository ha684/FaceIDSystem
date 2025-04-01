# Configuration settings for Face ID Attendance System

# Working hours configuration
WORK_START_TIME = '09:00'  # Regular work start time (24-hour format)
WORK_END_TIME = '17:00'    # Regular work end time (24-hour format)
LATE_THRESHOLD_MINUTES = 10  # Minutes after which an employee is considered late

# Paths
EMPLOYEE_DATABASE_DIR = 'employees'  # Directory containing employee face images
ATTENDANCE_RECORDS_DIR = 'attendance_records'  # Directory for storing attendance CSVs

# Face recognition settings
DETECTOR_BACKEND = 'retinaface'  # Options: opencv, ssd, dlib, mtcnn, retinaface, mediapipe
MODEL_NAME = 'Facenet512'  # Options: VGG-Face, Facenet, Facenet512, OpenFace, etc.
DISTANCE_METRIC = 'cosine'  # Options: cosine, euclidean, euclidean_l2
THRESHOLD = 0.4  # Lower is more strict (typical values: 0.3-0.5)

# Performance settings
USE_GPU = True  # Set to True to use GPU acceleration
GPU_MEMORY_FRACTION = 0.5  # Fraction of GPU memory to use (0.0 to 1.0)

# Sound settings
ENABLE_SOUND = True  # Set to True to enable sound notifications
SUCCESS_SOUND = 'success.mp3'  # Sound file for successful recognition
FAILURE_SOUND = 'failure.mp3'  # Sound file for failed recognition
MODE_SWITCH_SOUND = 'switch.mp3'  # Sound file for mode switching

# Video capture
CAMERA_ID = 0  # Default camera (0 is usually the built-in webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# UI Settings
FONT = 'Arial'
FONT_SCALE = 0.7
LINE_THICKNESS = 2
SHOW_DEBUG_INFO = True  # Show additional debug information
