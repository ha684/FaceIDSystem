from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Define parameters
backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8', 'yolov11s',
    'yolov11n', 'yolov11m', 'yunet', 'centerface',
]
detector = 'ssd'
align = True

# Path to your image
img_path = r"c:\Users\robot\Downloads\484644960_1032095908795469_4538832279473022526_n.jpg"

# Load the image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

# Extract faces
face_objs = DeepFace.extract_faces(
    img_path=img_path, 
    detector_backend=detector, 
    align=align,
)

# Create a copy of the image to draw on
img_with_bbox = img.copy()

# Draw bounding boxes and eye points
for face_obj in face_objs:
    facial_area = face_obj['facial_area']
    
    # Extract coordinates
    x = facial_area['x']
    y = facial_area['y']
    w = facial_area['w']
    h = facial_area['h']
    left_eye = facial_area['left_eye']
    right_eye = facial_area['right_eye']
    
    # Draw rectangle for face
    cv2.rectangle(img_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw circles for eyes
    cv2.circle(img_with_bbox, left_eye, 5, (255, 0, 0), -1)
    cv2.circle(img_with_bbox, right_eye, 5, (255, 0, 0), -1)
    
    # Add confidence score text
    confidence = face_obj['confidence']
    cv2.putText(img_with_bbox, f"Conf: {confidence:.2f}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with bounding boxes
plt.figure(figsize=(12, 12))
plt.imshow(img_with_bbox)
plt.axis('off')
plt.title('Detected Face with Bounding Box')
plt.show()

# If you prefer to use OpenCV window instead of matplotlib
def show_with_opencv():
    # Convert back to BGR for OpenCV display
    img_bgr = cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR)
    
    # Resize if the image is too large
    height, width = img_bgr.shape[:2]
    max_height = 800
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        img_bgr = cv2.resize(img_bgr, (new_width, max_height))
    
    # Show the image
    cv2.imshow("Detected Face", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uncomment to use OpenCV display instead of matplotlib
# show_with_opencv()

# Print information about the detected faces
for i, face_obj in enumerate(face_objs):
    print(f"Face {i+1}:")
    print(f"  Position: x={face_obj['facial_area']['x']}, y={face_obj['facial_area']['y']}")
    print(f"  Size: width={face_obj['facial_area']['w']}, height={face_obj['facial_area']['h']}")
    print(f"  Confidence: {face_obj['confidence']}")
    print(f"  Left eye position: {face_obj['facial_area']['left_eye']}")
    print(f"  Right eye position: {face_obj['facial_area']['right_eye']}")