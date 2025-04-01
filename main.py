import os
import cv2
import argparse
import tensorflow as tf
import numpy as np
import gradio as gr
import time
import datetime
from PIL import Image
import base64
import json
from face_recognition_module import FaceRecognitionSystem
from attendance_manager import AttendanceManager
from ui_manager import UIManager
from sound_manager import SoundManager
from registration_gui import launch_registration_gui
import config

ADMIN_PASSWORD = "admin123"

def register_employee_cli(employee_id, name, image_path):
    """Register a new employee to the system via command line."""
    face_recognition = FaceRecognitionSystem()
    result = face_recognition.register_employee(employee_id, name, image_path)
    if result['success']:
        print(f"\nEmployee {name} (ID: {employee_id}) registered successfully!")
    else:
        print(f"\nFailed to register employee {name}: {result['message']}")

def register_employee_gui():
    """Open the registration GUI for employee registration."""
    face_recognition = FaceRecognitionSystem()
    launch_registration_gui(face_recognition)

def generate_report(year=None, month=None):
    """Generate a monthly attendance report."""
    attendance_manager = AttendanceManager()
    report_path = attendance_manager.generate_monthly_report(year, month)
    print(f"\nMonthly report generated: {report_path}")

def check_gpu_status():
    """Check and display GPU status."""
    if config.USE_GPU:
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append(f"  - {gpu.name}")
                print(f"\nGPU Acceleration: Enabled")
                print(f"Found {len(gpus)} GPU(s):")
                for info in gpu_info:
                    print(info)
            else:
                print("\nGPU Acceleration: Requested but no GPU found. Using CPU.")
        except Exception as e:
            print(f"\nGPU Acceleration: Error initializing ({str(e)})")
    else:
        print("\nGPU Acceleration: Disabled")

def start_attendance_system():
    """Start the interactive face recognition attendance system."""
    try:
        print("\nInitializing Face ID Attendance System...")
        
        # Check GPU status
        check_gpu_status()
        
        # Initialize system components
        face_recognition = FaceRecognitionSystem()
        attendance_manager = AttendanceManager()
        ui_manager = UIManager(face_recognition, attendance_manager)
        sound_manager = SoundManager()
        
        print("Starting camera and UI...")
        ui_manager.start()

        print("\nSystem ready! Usage Instructions:\n"
            "- Face Recognition: Stand in front of the camera\n"
            "- Mode: Press 'M' to toggle between Check-In and Check-Out modes\n"
            "- Exit: Press 'ESC' to exit the system\n"
        )
        
        # Wait for the UI thread to finish (when user presses ESC)
        while ui_manager.is_running:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        ui_manager.stop()
        print("\nFace ID Attendance System shut down successfully.")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'ui_manager' in locals():
            ui_manager.stop()

def process_frame_for_recognition(frame, face_system, attendance_manager):
    """Process a single frame for face recognition and attendance tracking"""
    # Convert frame to RGB for DeepFace (it expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use the face recognition system to identify faces
    result = face_system.identify_face(rgb_frame)
    
    # Create a copy of the frame to draw on
    display_frame = frame.copy()
    
    # Get current time for display
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Default result message
    message = "No face detected"
    status_color = (200, 0, 0)  # Red for not recognized
    
    # Process the face detection result
    if 'face_region' in result and result['face_region'] is not None:
        x, y, x2, y2 = result['face_region']
        detection_confidence = result.get('detection_confidence', 0)
        
        # Different colors based on detection confidence
        if detection_confidence < 0.9:
            # Red box for low confidence detection
            color = (0, 0, 255)
            status_msg = f"Low confidence: {detection_confidence:.2f}"
        else:
            # Process face recognition if detection confidence is high enough
            if result['recognized']:
                # Green box for recognized face
                color = (0, 255, 0)
                name = result['name']
                confidence = result['confidence']
                
                # Record attendance
                if 'check_in_mode' in globals() and check_in_mode:
                    attendance_result = attendance_manager.record_check_in(result['employee_id'], name)
                    status_msg = f"Welcome {name}! Checked in at {datetime.datetime.now().strftime('%H:%M:%S')}"
                    status_color = (0, 200, 0)  # Green for success
                else:
                    attendance_result = attendance_manager.record_check_out(result['employee_id'], name)
                    status_msg = f"Goodbye {name}! Checked out at {datetime.datetime.now().strftime('%H:%M:%S')}"
                    status_color = (0, 200, 0)  # Green for success
                
                message = status_msg
            else:
                # Yellow box for unrecognized face with good detection
                color = (0, 255, 255)
                status_msg = "Face not recognized"
                message = "Unknown person detected"
                status_color = (0, 165, 255)  # Orange for unknown
        
        # Draw rectangle around the face
        cv2.rectangle(display_frame, (x, y), (x2, y2), color, 2)
        
        # Add detection confidence text
        conf_text = f"Conf: {detection_confidence:.2f}"
        cv2.putText(display_frame, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add the message at the bottom of the frame
    cv2.putText(display_frame, message, (10, display_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Add current time to the top of the frame
    cv2.putText(display_frame, f"Time: {current_time}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add mode indicator
    mode_text = "Mode: Check-In" if 'check_in_mode' in globals() and check_in_mode else "Mode: Check-Out"
    cv2.putText(display_frame, mode_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display_frame, message

def process_frame_for_registration(frame, face_system):
    """Process a single frame for face registration"""
    # Convert frame to RGB for DeepFace (it expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use RetinaFace to detect faces
    from retinaface import RetinaFace
    faces = RetinaFace.detect_faces(rgb_frame)
    
    # Create a copy of the frame to draw on
    display_frame = frame.copy()
    
    # Default message
    message = "No face detected"
    detected_face = None
    face_confidence = 0
    
    if isinstance(faces, dict):
        # Count the number of faces
        num_faces = len(faces)
        
        if num_faces == 0:
            message = "No face detected. Please position your face in front of the camera."
        elif num_faces > 1:
            message = "Multiple faces detected. Only one face should be visible for registration."
            # Draw rectangles around all faces
            for face_key in faces:
                face = faces[face_key]
                confidence = face["score"]
                facial_area = face["facial_area"]
                x, y, x2, y2 = facial_area
                
                # Use red color for warning about multiple faces
                cv2.rectangle(display_frame, (x, y), (x2, y2), (0, 0, 255), 2)
        else:
            # Single face - ideal for registration
            face_key = list(faces.keys())[0]
            face = faces[face_key]
            confidence = face["score"]
            facial_area = face["facial_area"]
            x, y, x2, y2 = facial_area
            
            if confidence > 0.9:
                # Good confidence - show green rectangle
                cv2.rectangle(display_frame, (x, y), (x2, y2), (0, 255, 0), 2)
                message = "Face detected with good confidence. You can register this face."
                
                # Extract the face region for registration
                detected_face = rgb_frame[y:y2, x:x2]
                face_confidence = confidence
            else:
                # Low confidence - show yellow rectangle
                cv2.rectangle(display_frame, (x, y), (x2, y2), (0, 255, 255), 2)
                message = f"Low confidence detection ({confidence:.2f}). Please adjust lighting or position."
    
    # Add the message at the bottom of the frame
    cv2.putText(display_frame, message, (10, display_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display_frame, message, detected_face, face_confidence

# Global variables for the Gradio app
face_recognition_system = None
attendance_manager = None
check_in_mode = True  # Default to check-in mode
registration_face = None
registration_confidence = 0

def authenticate(password):
    """Authenticate admin access"""
    if password == ADMIN_PASSWORD:
        return "Authentication successful! You can now access admin features."
    else:
        return "Authentication failed. Incorrect password."

def toggle_mode():
    """Toggle between check-in and check-out modes"""
    global check_in_mode
    check_in_mode = not check_in_mode
    return f"Switched to {'Check-In' if check_in_mode else 'Check-Out'} mode"

def registration_capture(frame):
    """Process frame for registration capture"""
    global registration_face, registration_confidence
    
    # Process the frame for face detection
    display_frame, message, detected_face, confidence = process_frame_for_registration(frame, face_recognition_system)
    
    # Store the detected face for registration
    registration_face = detected_face
    registration_confidence = confidence
    
    return display_frame, message

def register_face(employee_id, name):
    """Register a new face to the system"""
    global registration_face, registration_confidence
    
    if registration_face is None:
        return "No face has been captured. Please capture a face first."
    
    if registration_confidence < 0.9:
        return f"Face detection confidence too low ({registration_confidence:.2f}). Please recapture with better lighting/positioning."
    
    if not employee_id or not name:
        return "Employee ID and Name are required."
    
    try:
        # Save the face image temporarily
        temp_path = os.path.join("temp", f"{employee_id}.jpg")
        os.makedirs("temp", exist_ok=True)
        
        # Convert from RGB back to BGR for OpenCV
        bgr_face = cv2.cvtColor(registration_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_path, bgr_face)
        
        # Register the employee
        result = face_recognition_system.register_employee(employee_id, name, temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result['success']:
            # Reset the registration face
            registration_face = None
            registration_confidence = 0
            return f"Successfully registered {name} (ID: {employee_id})"
        else:
            return f"Registration failed: {result['message']}"
    
    except Exception as e:
        return f"Error during registration: {str(e)}"

def delete_face(password, employee_id, name):
    """Delete an employee's face data"""
    # First authenticate
    if password != ADMIN_PASSWORD:
        return "Authentication failed. Incorrect password."
    
    if not employee_id or not name:
        return "Employee ID and Name are required."
    
    try:
        # Check if employee directory exists
        employee_dir = os.path.join(config.EMPLOYEE_DATABASE_DIR, employee_id)
        if not os.path.exists(employee_dir):
            return f"No data found for employee ID: {employee_id}"
        
        # Check if the name matches
        face_file = os.path.join(employee_dir, f"{name}.jpg")
        if not os.path.exists(face_file):
            return f"No face data found for {name} with ID: {employee_id}"
        
        # Delete the employee data
        os.remove(face_file)
        
        # Remove directory if empty
        if len(os.listdir(employee_dir)) == 0:
            os.rmdir(employee_dir)
            
        return f"Successfully deleted face data for {name} (ID: {employee_id})"
    
    except Exception as e:
        return f"Error during deletion: {str(e)}"

def recognition_frame_processor(frame):
    """Process frames for the recognition tab"""
    global face_recognition_system, attendance_manager
    
    # Ensure the systems are initialized
    if face_recognition_system is None:
        face_recognition_system = FaceRecognitionSystem()
    
    if attendance_manager is None:
        attendance_manager = AttendanceManager()
    
    # Process the frame
    display_frame, message = process_frame_for_recognition(frame, face_recognition_system, attendance_manager)
    
    return display_frame, message

def launch_gradio_app():
    """Launch the Gradio web interface for the Face ID System"""
    global face_recognition_system, attendance_manager
    
    # Initialize the systems if not already done
    if face_recognition_system is None:
        face_recognition_system = FaceRecognitionSystem()
    
    if attendance_manager is None:
        attendance_manager = AttendanceManager()
    
    # Create the Gradio interface
    with gr.Blocks(title="Face ID Attendance System") as app:
        gr.Markdown("# Face ID Attendance System")
        gr.Markdown("A facial recognition based attendance system using RetinaFace")
        
        with gr.Tabs():
            # Recognition Tab
            with gr.TabItem("Recognition"):
                gr.Markdown("### Face Recognition and Attendance Tracking")
                
                with gr.Row():
                    with gr.Column():
                        recognition_camera = gr.Image(source="webcam", streaming=True, width=640, height=480)
                        mode_btn = gr.Button("Toggle Check-In/Check-Out Mode")
                    
                    with gr.Column():
                        recognition_status = gr.Textbox(label="Recognition Status", placeholder="Recognition status will appear here...")
                        time_display = gr.Textbox(label="Current Time", value=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        mode_display = gr.Textbox(label="Current Mode", value="Check-In Mode")
                
                # Set up the camera processing
                recognition_camera.stream(
                    recognition_frame_processor,
                    inputs=[recognition_camera],
                    outputs=[recognition_camera, recognition_status],
                    show_progress="hidden",
                )
                
                # Mode toggle button
                mode_btn.click(
                    toggle_mode,
                    inputs=[],
                    outputs=[mode_display]
                )
            
            # Registration Tab (Admin)
            with gr.TabItem("Registration"):
                gr.Markdown("### Employee Registration (Admin Only)")
                
                with gr.Row():
                    admin_password = gr.Textbox(type="password", label="Admin Password", placeholder="Enter admin password")
                    auth_btn = gr.Button("Authenticate")
                    auth_status = gr.Textbox(label="Authentication Status")
                
                auth_btn.click(
                    authenticate,
                    inputs=[admin_password],
                    outputs=[auth_status]
                )
                
                with gr.Row():
                    with gr.Column():
                        reg_camera = gr.Image(source="webcam", streaming=True, width=640, height=480)
                        capture_msg = gr.Textbox(label="Capture Status", placeholder="Face detection status will appear here...")
                    
                    with gr.Column():
                        employee_id = gr.Textbox(label="Employee ID", placeholder="Enter unique employee ID")
                        employee_name = gr.Textbox(label="Employee Name", placeholder="Enter employee's full name")
                        register_btn = gr.Button("Register Face")
                        register_status = gr.Textbox(label="Registration Status")
                
                # Set up the registration camera processing
                reg_camera.stream(
                    registration_capture,
                    inputs=[reg_camera],
                    outputs=[reg_camera, capture_msg],
                    show_progress="hidden",
                )
                
                # Register button functionality
                register_btn.click(
                    register_face,
                    inputs=[employee_id, employee_name],
                    outputs=[register_status]
                )
            
            # User Management Tab (Admin)
            with gr.TabItem("User Management"):
                gr.Markdown("### Delete Employee Data (Admin Only)")
                
                with gr.Row():
                    admin_del_password = gr.Textbox(type="password", label="Admin Password", placeholder="Enter admin password")
                    del_employee_id = gr.Textbox(label="Employee ID to Delete", placeholder="Enter employee ID")
                    del_employee_name = gr.Textbox(label="Employee Name to Delete", placeholder="Enter employee name")
                
                with gr.Row():
                    delete_btn = gr.Button("Delete Employee Data")
                    delete_status = gr.Textbox(label="Deletion Status")
                
                # Delete button functionality
                delete_btn.click(
                    delete_face,
                    inputs=[admin_del_password, del_employee_id, del_employee_name],
                    outputs=[delete_status]
                )
            
            # Reports Tab (Admin)
            with gr.TabItem("Reports"):
                gr.Markdown("### Attendance Reports (Admin Only)")
                
                with gr.Row():
                    admin_report_password = gr.Textbox(type="password", label="Admin Password", placeholder="Enter admin password")
                    report_year = gr.Number(label="Year (Optional)", placeholder="Leave empty for current year")
                    report_month = gr.Number(label="Month (Optional)", placeholder="Leave empty for current month")
                
                with gr.Row():
                    generate_report_btn = gr.Button("Generate Report")
                    report_status = gr.Textbox(label="Report Status")
                
                # Report generation functionality
                def generate_report_ui(password, year, month):
                    if password != ADMIN_PASSWORD:
                        return "Authentication failed. Incorrect password."
                    
                    attendance_manager = AttendanceManager()
                    try:
                        report_path = attendance_manager.generate_monthly_report(
                            int(year) if year else None,
                            int(month) if month else None
                        )
                        return f"Report generated successfully: {report_path}"
                    except Exception as e:
                        return f"Error generating report: {str(e)}"
                
                generate_report_btn.click(
                    generate_report_ui,
                    inputs=[admin_report_password, report_year, report_month],
                    outputs=[report_status]
                )
    
    # Launch the app
    app.launch(share=False)

def start_gradio_interface():
    """Start the Face ID System with Gradio interface"""
    try:
        print("\nInitializing Face ID Attendance System with Gradio interface...")
        
        # Check GPU status
        check_gpu_status()
        
        # Ensure required directories exist
        os.makedirs(config.EMPLOYEE_DATABASE_DIR, exist_ok=True)
        os.makedirs(config.ATTENDANCE_RECORDS_DIR, exist_ok=True)
        
        # Launch the Gradio interface
        launch_gradio_app()
    
    except Exception as e:
        print(f"\nError starting Gradio interface: {str(e)}")

def main():
    """Main function to parse command line arguments and run the system."""
    parser = argparse.ArgumentParser(description='Face ID Attendance System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Register employee command (CLI method)
    register_parser = subparsers.add_parser('register-cli', help='Register a new employee via command line')
    register_parser.add_argument('--id', required=True, help='Employee ID')
    register_parser.add_argument('--name', required=True, help='Employee name')
    register_parser.add_argument('--image', required=True, help='Path to employee face image')
    
    # Register employee with GUI
    subparsers.add_parser('register', help='Open the registration GUI for employee registration')
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate monthly attendance report')
    report_parser.add_argument('--year', type=int, help='Year for the report (default: current year)')
    report_parser.add_argument('--month', type=int, help='Month for the report (default: current month)')
    
    # Start attendance system command
    subparsers.add_parser('start', help='Start the attendance system')

    # Start Gradio interface command
    subparsers.add_parser('gradio', help='Start the system with Gradio web interface')
    
    args = parser.parse_args()
    
    if args.command == 'register-cli':
        register_employee_cli(args.id, args.name, args.image)
    elif args.command == 'register':
        register_employee_gui()
    elif args.command == 'report':
        generate_report(args.year, args.month)
    elif args.command == 'start':
        start_attendance_system()
    elif args.command == 'gradio':
        start_gradio_interface()
    else:
        # Default to Gradio interface if no command is specified
        start_gradio_interface()

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(config.EMPLOYEE_DATABASE_DIR, exist_ok=True)
    os.makedirs(config.ATTENDANCE_RECORDS_DIR, exist_ok=True)
    
    main()
