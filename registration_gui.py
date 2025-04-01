import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import threading
import os
import time
from PIL import Image, ImageTk
import config
from sound_manager import SoundManager

class RegistrationGUI:
    """GUI for employee registration process with face detection and progress indicators."""
    
    def __init__(self, face_recognition_system):
        """Initialize the registration GUI.
        
        Args:
            face_recognition_system: Instance of FaceRecognitionSystem
        """
        self.face_recognition = face_recognition_system
        self.sound_manager = SoundManager()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Employee Registration")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Set style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        self.style.configure("Success.TLabel", foreground="green")
        self.style.configure("Error.TLabel", foreground="red")
        
        # Create frame layout
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_label = ttk.Label(self.main_frame, text="Employee Registration", style="Header.TLabel")
        header_label.grid(row=0, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Form fields
        ttk.Label(self.main_frame, text="Employee ID:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.employee_id_var = tk.StringVar()
        employee_id_entry = ttk.Entry(self.main_frame, textvariable=self.employee_id_var, width=30)
        employee_id_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(self.main_frame, text="Full Name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(self.main_frame, textvariable=self.name_var, width=30)
        name_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Image selection and preview
        ttk.Label(self.main_frame, text="Face Image:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.file_path_var = tk.StringVar()
        file_path_entry = ttk.Entry(self.main_frame, textvariable=self.file_path_var, width=30)
        file_path_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        browse_button = ttk.Button(self.main_frame, text="Browse...", command=self.browse_image)
        browse_button.grid(row=3, column=2, sticky=tk.W, pady=5)
        
        # Webcam capture button
        webcam_button = ttk.Button(self.main_frame, text="Use Webcam", command=self.open_webcam)
        webcam_button.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Image preview area
        preview_frame = ttk.Frame(self.main_frame, width=320, height=240, borderwidth=2, relief=tk.GROOVE)
        preview_frame.grid(row=5, column=0, columnspan=3, pady=10)
        preview_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Progress area
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky=tk.W+tk.E)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=tk.W+tk.E)
        
        self.status_var = tk.StringVar()
        status_label = ttk.Label(self.progress_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)
        
        register_button = ttk.Button(button_frame, text="Register Employee", command=self.register_employee)
        register_button.pack(side=tk.LEFT, padx=5)
        
        close_button = ttk.Button(button_frame, text="Close", command=self.on_close)
        close_button.pack(side=tk.LEFT, padx=5)
        
        # Instance variables
        self.image = None
        self.webcam = None
        self.is_webcam_open = False
        self.capture_thread = None
        self.processing = False
        
        # Center the window
        self.center_window()
    
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def browse_image(self):
        """Open file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.load_preview_image(file_path)
    
    def load_preview_image(self, image_path):
        """Load and display the selected image."""
        try:
            # Close webcam if open
            self.close_webcam()
            
            # Load image using PIL
            pil_image = Image.open(image_path)
            
            # Resize to fit preview area
            pil_image.thumbnail((320, 240))
            
            # Convert to PhotoImage
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update preview label
            self.preview_label.configure(image=tk_image)
            self.preview_label.image = tk_image  # Keep a reference
            
            # Store the actual image path
            self.image = image_path
            
            # Run face detection
            self.detect_faces_in_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def detect_faces_in_preview(self):
        """Detect faces in the preview image and highlight them."""
        if not self.image:
            return
            
        threading.Thread(target=self._detect_faces_thread).start()
    
    def _detect_faces_thread(self):
        """Background thread for face detection."""
        try:
            self.status_var.set("Detecting faces...")
            self.progress_var.set(30)
            
            # Load image with OpenCV
            img = cv2.imread(self.image)
            
            # Extract faces
            face_objs = self.face_recognition.face_recognition.extract_faces(
                img_path=self.image,
                detector_backend=self.face_recognition.detector_backend,
                align=True
            )
            
            self.progress_var.set(70)
            
            # Create a copy for drawing
            img_draw = img.copy()
            
            if len(face_objs) == 0:
                self.status_var.set("No faces detected. Please select another image.")
                self.sound_manager.play_failure()
            elif len(face_objs) > 1:
                # Draw multiple faces with warning
                for i, face_obj in enumerate(face_objs):
                    region = face_obj.get('facial_area', {})
                    if region:
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(img_draw, f"Face {i+1}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                self.status_var.set(f"Warning: {len(face_objs)} faces detected. Please use an image with a single face.")
                self.sound_manager.play_failure()
            else:
                # Single face - draw in green
                face_obj = face_objs[0]
                region = face_obj.get('facial_area', {})
                if region:
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img_draw, "Face detected", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                self.status_var.set("Face detected successfully")
                self.sound_manager.play_success()
            
            # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and resize
            pil_image = Image.fromarray(img_rgb)
            pil_image.thumbnail((320, 240))
            
            # Convert to PhotoImage
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update preview label on main thread
            self.root.after(0, lambda: self._update_preview(tk_image))
            
            self.progress_var.set(100)
        
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.progress_var.set(0)
    
    def _update_preview(self, tk_image):
        """Update the preview label with the new image."""
        self.preview_label.configure(image=tk_image)
        self.preview_label.image = tk_image  # Keep a reference
    
    def open_webcam(self):
        """Open webcam for live capture."""
        if self.is_webcam_open:
            return
        
        # Initialize webcam
        self.webcam = cv2.VideoCapture(config.CAMERA_ID)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.webcam.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        
        self.is_webcam_open = True
        self.status_var.set("Webcam active. Click 'Capture' when ready.")
        
        # Add capture button
        self.capture_button = ttk.Button(self.main_frame, text="Capture", command=self.capture_image)
        self.capture_button.grid(row=4, column=2, sticky=tk.W, pady=5)
        
        # Start webcam thread
        self.capture_thread = threading.Thread(target=self.update_webcam_feed)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def update_webcam_feed(self):
        """Update the webcam feed in the preview area."""
        while self.is_webcam_open:
            ret, frame = self.webcam.read()
            if ret:
                # Convert to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                pil_image.thumbnail((320, 240))
                
                # Convert to PhotoImage
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # Update preview on main thread
                self.root.after(0, lambda img=tk_image: self._update_preview(img))
            
            time.sleep(0.03)  # About 30 FPS
    
    def capture_image(self):
        """Capture an image from the webcam."""
        if not self.is_webcam_open:
            return
        
        ret, frame = self.webcam.read()
        if ret:
            # Create temp directory if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save the captured image
            temp_file = os.path.join(temp_dir, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(temp_file, frame)
            
            # Update the file path
            self.file_path_var.set(temp_file)
            
            # Close webcam
            self.close_webcam()
            
            # Load the captured image for preview
            self.load_preview_image(temp_file)
        else:
            messagebox.showerror("Error", "Failed to capture image from webcam")
    
    def close_webcam(self):
        """Close the webcam."""
        if self.is_webcam_open:
            self.is_webcam_open = False
            if self.webcam:
                self.webcam.release()
            
            if hasattr(self, 'capture_button'):
                self.capture_button.grid_forget()
    
    def register_employee(self):
        """Register an employee with the provided details."""
        # Check if already processing
        if self.processing:
            return
            
        # Get form data
        employee_id = self.employee_id_var.get().strip()
        name = self.name_var.get().strip()
        image_path = self.file_path_var.get().strip()
        
        # Validate inputs
        if not employee_id:
            messagebox.showerror("Error", "Employee ID is required")
            return
        
        if not name:
            messagebox.showerror("Error", "Name is required")
            return
        
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image")
            return
        
        # Set processing flag
        self.processing = True
        
        # Reset progress
        self.progress_var.set(0)
        self.status_var.set("Registering employee...")
        
        # Start registration in a separate thread
        threading.Thread(target=self._register_thread, args=(employee_id, name, image_path)).start()
    
    def _register_thread(self, employee_id, name, image_path):
        """Background thread for employee registration."""
        try:
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(20))
            
            # Register employee
            result = self.face_recognition.register_employee(employee_id, name, image_path)
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(80))
            
            # Process result
            if result['success']:
                self.root.after(0, lambda: self._registration_success(name))
                self.sound_manager.play_success()
            else:
                error_msg = result.get('message', "Unknown error")
                self.root.after(0, lambda msg=error_msg: self._registration_failed(msg))
                self.sound_manager.play_failure()
                
                # If multiple faces were detected, show a more helpful message
                if result.get('status') == 'multiple_faces_detected':
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Multiple Faces", 
                        "Multiple faces were detected in the image. Please use an image with only one face."
                    ))
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(100))
            
        except Exception as e:
            self.root.after(0, lambda err=str(e): self._registration_failed(f"Error: {err}"))
            self.sound_manager.play_failure()
        finally:
            # Clear processing flag
            self.processing = False
    
    def _registration_success(self, name):
        """Handle successful registration."""
        self.status_var.set(f"Employee '{name}' registered successfully")
        messagebox.showinfo("Success", f"Employee '{name}' has been registered successfully.")
        
        # Clear form
        self.employee_id_var.set("")
        self.name_var.set("")
        self.file_path_var.set("")
        
        # Clear preview
        self.preview_label.configure(image=None)
        self.preview_label.image = None
        self.image = None
    
    def _registration_failed(self, error_message):
        """Handle failed registration."""
        self.status_var.set(f"Registration failed: {error_message}")
        messagebox.showerror("Registration Failed", error_message)
    
    def on_close(self):
        """Handle window close event."""
        self.close_webcam()
        self.root.destroy()
    
    def run(self):
        """Run the application main loop."""
        self.root.mainloop()


# Function to launch the registration GUI
def launch_registration_gui(face_recognition_system):
    """Launch the employee registration GUI.
    
    Args:
        face_recognition_system: Instance of FaceRecognitionSystem
    """
    app = RegistrationGUI(face_recognition_system)
    app.run()
