import os
import cv2
import time
import numpy as np
import gradio as gr
import threading
import json
from face_recognition_module import FaceRecognitionSystem
from attendance_manager import AttendanceManager

# Global variables
current_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()
cap = None

ADMIN_PASSWORD = "123456"  # hash for "password"
# State management
class SystemState:
    def __init__(self):
        self.is_admin_mode = False
        self.is_add_face_mode = False
        self.is_delete_face_mode = False
        self.detection_only = True
        self.last_recognized_name = None
        self.last_recognition_time = None
        self.camera_active = False
        self.last_valid_frame = None
        
system_state = SystemState()
face_recognition = FaceRecognitionSystem()
attendance_manager = AttendanceManager()
employees = attendance_manager.load_employee_database()

# Thread management functions
def capture_frames():
    global current_frame, cap
    cap = cv2.VideoCapture(0)
    
    # Try to set camera to high frame rate
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS if camera supports it
    
    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            break
        
        # Convert BGR to RGB for Gradio display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame immediately
        processed_frame = process_frame_for_display(frame)
        
        # Update current frame with thread safety
        with frame_lock:
            current_frame = processed_frame
        
    if cap and cap.isOpened():
        cap.release()

def process_frame_for_display(frame):
    """Process frame for display purposes only, returns just the frame"""
    if frame is None:
        # Return a blank frame with message if camera is off
        blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
        cv2.putText(blank_frame, "Camera Off", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return blank_frame
    
    output_frame = frame.copy()
    faces = face_recognition.detect_faces(frame)
    
    # Process detected faces
    for face_key in faces:
        face = faces[face_key]
        facial_area = face["facial_area"]
        x, y, w, h = facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[1]
        confidence = face['score']
        
        if confidence >= face_recognition.FACE_DETECTION_THRESHOLD:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
        confidence_text = f"{confidence:.2f}"
        cv2.putText(output_frame, confidence_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output_frame

def start_cam():
    if system_state.camera_active:
        return 'Camera already running'
        
    stop_event.clear()
    system_state.camera_active = True
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    return 'Camera started'

def stop_cam():
    system_state.camera_active = False
    stop_event.set()
    global cap
    if cap and cap.isOpened():
        cap.release()
    # Clear the current frame
    with frame_lock:
        global current_frame
        current_frame = None
    return 'Camera stopped'

def update_display():
    if system_state.camera_active:
        with frame_lock:
            if current_frame is not None:
                return current_frame, "Camera Active", "Monitoring for faces"
    
    # Return a blank frame with message if camera is off or no frame
    blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
    cv2.putText(blank_frame, "Camera Off", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return blank_frame, "Camera Off", "Press 'Turn On Camera' to start"

def process_frame(frame, admin_password="", employee_name="", employee_id=""):
    if frame is None or not system_state.camera_active:
        # Return a blank frame with message if camera is off
        blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
        cv2.putText(blank_frame, "Camera Off", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return blank_frame, "Camera Off", "Press 'Turn On Camera' to start"
    
    output_frame = frame.copy()
    
    if admin_password and not system_state.is_admin_mode:
        if admin_password == ADMIN_PASSWORD:
            system_state.is_admin_mode = True
            return frame, "Admin Mode Activated", "You can now add or delete faces."
        else:
            return frame, "Authentication Failed", "Incorrect password."
    
    faces = face_recognition.detect_faces(frame)
    if not faces:
        return output_frame, "No Faces Detected", "Please position face in frame."
    
    # Process detected faces
    for face_key in faces:
        face = faces[face_key]
        facial_area = face["facial_area"]
        
        # Fixed: Properly access the coordinates from the facial_area list
        x, y = facial_area[0], facial_area[1]
        w, h = facial_area[2] - facial_area[0], facial_area[3] - facial_area[1]
        confidence = face['score']
        
        if confidence >= face_recognition.FACE_DETECTION_THRESHOLD:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
        confidence_text = f"{confidence:.2f}"
        cv2.putText(output_frame, confidence_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Admin mode: Add face
    if system_state.is_admin_mode and system_state.is_add_face_mode:
        if len(faces) > 1:
            return output_frame, "Multiple Faces Detected", "Only one face allowed for registration. Please ensure only one person is in frame."
        
        if len(faces) == 1 and list(faces.values())[0]['score'] >= face_recognition.FACE_DETECTION_THRESHOLD:
            if employee_name and employee_id:
                employee_img_path = os.path.join(attendance_manager.records_dir, f"{employee_id}_{employee_name}.jpg")
                cv2.imwrite(employee_img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                employees[employee_id] = {"id": employee_id, "name": employee_name}
                attendance_manager.save_employee_database(employees)
                
                system_state.is_add_face_mode = False
                
                return output_frame, "Face Registered", f"Employee {employee_name} (ID: {employee_id}) has been registered."
            else:
                return output_frame, "Ready to Register", "Face detected. Please enter name and ID to register."
        
        return output_frame, "Add Face Mode", "Position face in frame, ensure >90% confidence, then enter name and ID."
    
    # Admin mode: Delete face
    if system_state.is_admin_mode and system_state.is_delete_face_mode:
        if employee_id:
            if employee_id in employees:
                employee_name = employees[employee_id]["name"]
                del employees[employee_id]
                attendance_manager.save_employee_database(employees)
                
                employee_img_path = os.path.join(attendance_manager.records_dir, f"{employee_id}_{employee_name}.jpg")
                if os.path.exists(employee_img_path):
                    os.remove(employee_img_path)
                
                # Reset state
                system_state.is_delete_face_mode = False
                
                return output_frame, "Face Deleted", f"Employee {employee_name} (ID: {employee_id}) has been removed."
            else:
                return output_frame, "Error", f"Employee with ID {employee_id} not found."
        
        return output_frame, "Delete Face Mode", "Enter employee ID to delete."
    
    # Normal recognition mode
    if len(faces) > 0 and not system_state.is_admin_mode:
        # Find the face with highest confidence
        best_face_key = max(faces, key=lambda k: faces[k].get('score', 0))
        best_face = faces[best_face_key]
        
        if best_face['score'] >= face_recognition.FACE_DETECTION_THRESHOLD:
            employee_info, confidence = face_recognition.recognize_face(frame, employees)
            if employee_info:
                status, time_recorded = attendance_manager.record_attendance(employee_info["id"], employee_info["name"])
                
                # Fixed: Properly access the facial area coordinates
                facial_area = best_face['facial_area']
                x, y = facial_area[0], facial_area[1]
                
                system_state.last_recognized_name = employee_info["name"]
                system_state.last_recognition_time = time.time()
                
                name_status = f"{employee_info['name']} - {status}"
                cv2.putText(output_frame, name_status, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return output_frame, f"Employee Recognized: {employee_info['name']}", f"{status} recorded at {time_recorded}"
            else:
                return output_frame, "Face Detected", "Unknown person. Not in employee database."
        else:
            return output_frame, "Low Confidence Detection", "Please adjust position for better face detection."
    
    if system_state.is_admin_mode:
        return output_frame, "Admin Mode Active", "Use buttons below to add or delete faces."
    else:
        return output_frame, "Attendance System Active", "Position face in frame for recognition."

# Button click handlers
def toggle_add_face_mode():
    if not system_state.is_admin_mode:
        return "Authentication Required", "Please enter admin password first."
    
    system_state.is_add_face_mode = True
    system_state.is_delete_face_mode = False
    return "Add Face Mode Activated", "Position one face in frame and enter details."

def toggle_delete_face_mode():
    if not system_state.is_admin_mode:
        return "Authentication Required", "Please enter admin password first."
    
    system_state.is_delete_face_mode = True
    system_state.is_add_face_mode = False
    return "Delete Face Mode Activated", "Enter employee ID to delete."

def exit_admin_mode():
    system_state.is_admin_mode = False
    system_state.is_add_face_mode = False
    system_state.is_delete_face_mode = False
    return "Admin Mode Deactivated", "System returned to attendance mode."

def process_camera_frame():
    if system_state.camera_active:
        with frame_lock:
            if current_frame is not None:
                return current_frame
    
    # Return a blank frame if camera is off or no frame available
    blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
    cv2.putText(blank_frame, "Camera Off", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return blank_frame

with gr.Blocks(title="Face Recognition Attendance System") as demo:
    gr.Markdown("# Face Recognition Attendance System")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Camera display (NOT using Gradio's webcam sources)
            camera_feed = gr.Image(label="Camera Feed", image_mode='RGB')
            
            # Camera control buttons
            with gr.Row():
                camera_on_btn = gr.Button("Turn On Camera", variant="primary")
                camera_off_btn = gr.Button("Turn Off Camera", variant="stop")
        
        with gr.Column(scale=1):
            # Status displays
            status_text = gr.Textbox(label="Status", value="System Ready")
            info_text = gr.Textbox(label="Information", value="Press 'Turn On Camera' to start")
            
            # Admin authentication
            gr.Markdown("### Admin Authentication")
            admin_password = gr.Textbox(label="Admin Password", type="password")
            auth_btn = gr.Button("Authenticate")
            
            # Employee registration inputs
            gr.Markdown("### Employee Registration")
            employee_name = gr.Textbox(label="Employee Name")
            employee_id = gr.Textbox(label="Employee ID")
            
            # Admin action buttons
            gr.Markdown("### Admin Actions")
            with gr.Row():
                add_face_btn = gr.Button("Add Face")
                delete_face_btn = gr.Button("Delete Face")
                exit_admin_btn = gr.Button("Exit Admin Mode")
            
            # System information
            gr.Markdown("### System Information")
            view_employees_btn = gr.Button("View Employees")
            employees_text = gr.Textbox(label="Employee List", interactive=False)
            
            view_attendance_btn = gr.Button("View Today's Attendance")
            attendance_text = gr.Textbox(label="Attendance Records", interactive=False)
    
    # Use a faster timer for more responsive updates
    timer = gr.Timer(0.01)  # 100 Hz refresh rate
    timer.tick(process_camera_frame, outputs=[camera_feed])

    # Update status after camera operation
    camera_on_btn.click(
        fn=start_cam,
        inputs=None,
        outputs=info_text
    ).then(
        fn=lambda: "Camera Active",
        inputs=None,
        outputs=status_text
    )
    
    camera_off_btn.click(
        fn=stop_cam,
        inputs=None,
        outputs=info_text
    ).then(
        fn=lambda: "Camera Off",
        inputs=None,
        outputs=status_text
    )
    
    # Process and recognize faces in regular mode or admin mode
    def process_current_frame(password, name, id):
        with frame_lock:
            if current_frame is not None:
                return process_frame(current_frame, password, name, id)
            else:
                return process_frame(None, password, name, id)
    
    # Admin authentication
    auth_btn.click(
        fn=process_current_frame,
        inputs=[admin_password, employee_name, employee_id],
        outputs=[camera_feed, status_text, info_text]
    )
    
    # Admin action event handlers
    add_face_btn.click(
        fn=toggle_add_face_mode,
        inputs=[],
        outputs=[status_text, info_text]
    )
    
    delete_face_btn.click(
        fn=toggle_delete_face_mode,
        inputs=[],
        outputs=[status_text, info_text]
    )
    
    exit_admin_btn.click(
        fn=exit_admin_mode,
        inputs=[],
        outputs=[status_text, info_text]
    )
    
    # Process registration when employee name and ID are submitted
    def submit_employee_registration():
        with frame_lock:
            if current_frame is not None:
                return process_frame(current_frame, "", employee_name.value, employee_id.value)
            else:
                return process_frame(None, "", employee_name.value, employee_id.value)
    
    # Add a submit button for employee registration
    register_btn = gr.Button("Register Employee")
    register_btn.click(
        fn=submit_employee_registration,
        inputs=[],
        outputs=[camera_feed, status_text, info_text]
    )
    
    view_employees_btn.click(
        fn=lambda: json.dumps(attendance_manager.load_employee_database(), indent=2),
        inputs=[],
        outputs=[employees_text]
    )
    
    view_attendance_btn.click(
        fn=lambda: json.dumps(attendance_manager.get_today_attendance(), indent=2),
        inputs=[],
        outputs=[attendance_text]
    )

# Launch the Gradio app
if __name__ == "__main__":
    print("Starting Face Recognition Attendance System...")
    # Make sure the record directory exists
    os.makedirs(attendance_manager.records_dir, exist_ok=True)
    demo.launch(share=False)