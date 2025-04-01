import cv2
import numpy as np
import time
import threading
import config
from datetime import datetime
from sound_manager import SoundManager

class UIManager:
    """Class to manage the user interface for the Face ID Attendance System."""

    def __init__(self, face_recognition_system, attendance_manager):
        """Initialize the UI manager.

        Args:
            face_recognition_system: Instance of FaceRecognitionSystem
            attendance_manager: Instance of AttendanceManager
        """
        self.face_recognition = face_recognition_system
        self.attendance_manager = attendance_manager
        
        # Initialize sound manager
        self.sound_manager = SoundManager()
        
        # Initialize camera
        self.camera_id = config.CAMERA_ID
        self.frame_width = config.FRAME_WIDTH
        self.frame_height = config.FRAME_HEIGHT
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.FONT_SCALE
        self.line_thickness = config.LINE_THICKNESS
        self.show_debug = config.SHOW_DEBUG_INFO
        
        # State variables
        self.is_running = False
        self.current_frame = None
        self.recognition_results = None
        self.last_recognized_time = 0
        self.recognition_cooldown = 2  # seconds (reduced from 3 for faster response)
        self.check_in_mode = True  # True = check-in mode, False = check-out mode
        self.status_message = "Initializing system..."
        self.status_color = (200, 200, 200)  # Light gray
        
        # Face detection box colors
        self.colors = {
            'recognized': (0, 255, 0),     # Green
            'unrecognized': (0, 0, 255),   # Red
            'processing': (255, 165, 0),   # Orange
            'mode_checkin': (0, 255, 0),   # Green
            'mode_checkout': (0, 165, 255), # Orange
            'background': (50, 50, 50),    # Dark gray
            'text': (255, 255, 255)        # White
        }
        
        # Processing flag to indicate when recognition is in progress
        self.is_processing = False
        
    def toggle_check_mode(self):
        """Toggle between check-in and check-out mode."""
        self.check_in_mode = not self.check_in_mode
        mode_str = "Check-In" if self.check_in_mode else "Check-Out"
        self.status_message = f"Switched to {mode_str} Mode"
        self.status_color = self.colors['mode_checkin'] if self.check_in_mode else self.colors['mode_checkout']
        
        # Play switch sound
        self.sound_manager.play_switch()
        
    def start(self):
        """Start the UI manager and video capture."""
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            raise Exception("Could not open video capture device")
        
        # Start video processing thread
        self.is_running = True
        self.thread = threading.Thread(target=self._process_video)
        self.thread.daemon = True
        self.thread.start()
        
        self.status_message = "System Ready - Press 'M' to toggle mode"
        self.status_color = self.colors['mode_checkin']
        
    def stop(self):
        """Stop the UI manager and release resources."""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
    
    def _process_video(self):
        """Process video frames and perform face recognition."""
        while self.is_running:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.status_message = "Error: Could not read from camera"
                self.status_color = self.colors['unrecognized']
                time.sleep(0.1)
                continue
            
            # Make a copy of the frame for display
            display_frame = frame.copy()
            
            # Perform face recognition if cooldown period has passed and not already processing
            current_time = time.time()
            if (current_time - self.last_recognized_time > self.recognition_cooldown) and not self.is_processing:
                # Set processing flag
                self.is_processing = True
                
                # Run face recognition in a separate thread to avoid UI freezing
                threading.Thread(
                    target=self._recognize_face_thread, 
                    args=(frame.copy(),)
                ).start()
            
            # Draw UI elements on the display frame
            self._draw_ui(display_frame)
            
            # Update the current frame
            self.current_frame = display_frame
            
            # Display the frame
            cv2.imshow('Face ID Attendance System', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                self.is_running = False
            elif key == ord('m') or key == ord('M'):  # 'M' key to toggle mode
                self.toggle_check_mode()
                
    def _recognize_face_thread(self, frame):
        """Perform face recognition in a separate thread.

        Args:
            frame: The video frame to process
        """
        try:
            # Update status to show processing
            self.status_message = "Processing..."
            self.status_color = self.colors['processing']
            
            # Check for anti-spoofing
            anti_spoof_result = self.face_recognition.verify_anti_spoofing(frame)
            if not anti_spoof_result['is_real']:
                self.recognition_results = {
                    'recognized': False,
                    'message': 'Spoof detected! Please use a real face.',
                    'status': 'spoof_detected'
                }
                self.status_message = "Spoof detected!"
                self.status_color = self.colors['unrecognized']
                self.sound_manager.play_failure()
                self.last_recognized_time = time.time()
                self.is_processing = False
                return
            
            # Perform face recognition
            recognition_result = self.face_recognition.identify_face(frame)
            self.recognition_results = recognition_result
            
            # If face is recognized, record attendance
            if recognition_result and recognition_result.get('recognized', False):
                employee_id = recognition_result['employee_id']
                name = recognition_result['name']
                
                # Record check-in or check-out based on current mode
                if self.check_in_mode:
                    result = self.attendance_manager.record_check_in(employee_id, name)
                else:
                    result = self.attendance_manager.record_check_out(employee_id, name)
                
                if result['success']:
                    if self.check_in_mode:
                        self.status_message = f"Check-in: {name}, Status: {result.get('status', '')}"
                    else:
                        self.status_message = f"Check-out: {name}, Duration: {result.get('duration', '')}"
                    self.status_color = self.colors['recognized']
                    self.sound_manager.play_success()
                else:
                    self.status_message = result['message']
                    self.status_color = self.colors['processing']
                    # Not a complete failure, so no failure sound
            else:
                message = "Face not recognized" 
                if recognition_result and 'message' in recognition_result:
                    message = recognition_result['message']
                self.status_message = message
                self.status_color = self.colors['unrecognized']
                self.sound_manager.play_failure()
            
            self.last_recognized_time = time.time()
        
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            self.status_color = self.colors['unrecognized']
            self.sound_manager.play_failure()
            self.last_recognized_time = time.time()
        
        finally:
            # Clear processing flag
            self.is_processing = False
    
    def _draw_ui(self, frame):
        """Draw UI elements on the frame.

        Args:
            frame: The video frame to draw on
        """
        # Draw top banner with current time and date
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), self.colors['background'], -1)
        cv2.putText(frame, f"Date: {current_date} | Time: {current_time}", 
                    (10, 30), self.font, self.font_scale, self.colors['text'], self.line_thickness)
        
        # Draw mode indicator with distinct colors
        mode_text = "CHECK-IN MODE" if self.check_in_mode else "CHECK-OUT MODE"
        mode_color = self.colors['mode_checkin'] if self.check_in_mode else self.colors['mode_checkout']
        cv2.putText(frame, mode_text, (frame.shape[1] - 200, 30), 
                    self.font, self.font_scale, mode_color, self.line_thickness)
        
        # Draw face detection box and name if face is detected
        if self.recognition_results:
            # Draw based on recognition status
            if 'face_region' in self.recognition_results and self.recognition_results['face_region']:
                # Get face region coordinates
                region = self.recognition_results['face_region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Choose color based on recognition status
                if self.is_processing:
                    box_color = self.colors['processing']
                    text = "Processing..."
                elif self.recognition_results.get('recognized', False):
                    box_color = self.colors['recognized']
                    name = self.recognition_results['name']
                    confidence = self.recognition_results.get('confidence', 0)
                    text = f"{name} ({confidence:.2f})"
                else:
                    box_color = self.colors['unrecognized']
                    text = "Unknown"
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                
                # Draw text background for better visibility
                text_size = cv2.getTextSize(text, self.font, self.font_scale, self.line_thickness)[0]
                cv2.rectangle(frame, (x, y-text_size[1]-10), (x+text_size[0], y), box_color, -1)
                
                # Draw name and confidence
                cv2.putText(frame, text, (x, y-5), 
                            self.font, self.font_scale, self.colors['text'], self.line_thickness)
        
        # Draw processing indicator if recognition is in progress
        if self.is_processing:
            # Draw a spinner or indicator to show processing
            cv2.putText(frame, "Processing...", (frame.shape[1]//2 - 70, 70), 
                        self.font, self.font_scale, self.colors['processing'], self.line_thickness)
        
        # Draw status message at the bottom
        cv2.rectangle(frame, (0, frame.shape[0]-40), (frame.shape[1], frame.shape[0]), self.colors['background'], -1)
        cv2.putText(frame, self.status_message, (10, frame.shape[0]-15), 
                    self.font, self.font_scale, self.status_color, self.line_thickness)
        
        # Draw instructions
        instructions = "Press 'M' to toggle Check-In/Check-Out mode | Press 'ESC' to exit"
        cv2.putText(frame, instructions, (10, frame.shape[0]-60), 
                    self.font, 0.5, (200, 200, 200), 1)
        
        # Draw debug information if enabled
        if self.show_debug and self.recognition_results:
            # Add detailed debug info in top-left corner
            debug_y = 80
            for key, value in self.recognition_results.items():
                if key not in ['face_region', 'face'] and value is not None:
                    # Skip complex objects to avoid cluttering the display
                    if isinstance(value, dict) or isinstance(value, list):
                        continue
                    
                    debug_text = f"{key}: {value}"
                    cv2.putText(frame, debug_text, (10, debug_y), 
                                self.font, 0.4, (255, 255, 255), 1)
                    debug_y += 20
