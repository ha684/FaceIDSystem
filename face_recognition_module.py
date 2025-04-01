import os
import cv2
import numpy as np
from deepface import DeepFace
import config
import tensorflow as tf


class FaceRecognitionSystem:
    """Class to handle face recognition operations for the attendance system."""

    def __init__(self):
        """Initialize the face recognition system."""
        self.model_name = config.MODEL_NAME
        self.detector_backend = config.DETECTOR_BACKEND
        self.distance_metric = config.DISTANCE_METRIC
        self.threshold = config.THRESHOLD
        self.db_path = config.EMPLOYEE_DATABASE_DIR

        # Configure GPU if available and requested
        self._configure_gpu()

        # Ensure the employee database directory exists
        os.makedirs(self.db_path, exist_ok=True)

        # Initialize employee database
        self.employee_db = {}
        
    def _configure_gpu(self):
        """Configure GPU settings for TensorFlow."""
        if config.USE_GPU:
            try:
                # Check if GPU is available
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    # Set memory growth to avoid taking all GPU memory
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Limit GPU memory usage if specified
                    if config.GPU_MEMORY_FRACTION < 1.0:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpus[0],
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=int(config.GPU_MEMORY_FRACTION * 1024 * 10)
                            )]
                        )
                    print(f"GPU acceleration enabled with {len(gpus)} device(s)")
                else:
                    print("No GPU found, falling back to CPU")
            except Exception as e:
                print(f"Error configuring GPU: {e}")
                print("Falling back to CPU")

    def register_employee(self, employee_id, name, image_path):
        """Register a new employee to the face recognition system.

        Args:
            employee_id (str): Unique identifier for the employee
            name (str): Full name of the employee
            image_path (str): Path to the employee's face image

        Returns:
            dict: Registration status and details
        """
        try:
            # Create employee directory if it doesn't exist
            employee_dir = os.path.join(self.db_path, employee_id)
            os.makedirs(employee_dir, exist_ok=True)
            
            # Extract faces from the image
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                align=True
            )
            
            # Check if any face was detected
            if len(face_objs) == 0:
                return {
                    'success': False,
                    'message': f"No face detected in the image for employee {name}",
                    'status': 'no_face_detected'
                }
            
            # Check if multiple faces were detected
            if len(face_objs) > 1:
                return {
                    'success': False,
                    'message': f"Multiple faces detected in the image. Please use an image with only one face.",
                    'status': 'multiple_faces_detected',
                    'face_count': len(face_objs)
                }
            
            # Save the face image to employee directory
            new_image_path = os.path.join(employee_dir, f"{name}.jpg")
            
            # Get the detected face
            detected_face = face_objs[0]['face']
            cv2.imwrite(new_image_path, detected_face)
            
            # Store employee info
            self.employee_db[employee_id] = {
                'name': name,
                'image_path': new_image_path
            }
            
            print(f"Employee {name} registered successfully")
            return {
                'success': True,
                'message': f"Employee {name} registered successfully",
                'status': 'registered',
                'image_path': new_image_path,
                'face': detected_face
            }
        
        except Exception as e:
            print(f"Error registering employee: {e}")
            return {
                'success': False,
                'message': f"Error registering employee: {str(e)}",
                'status': 'error'
            }
            
    def identify_face(self, image):
        """Identify a face in the given image against the employee database.

        Args:
            image: Either a path to an image file or a numpy array containing the image

        Returns:
            dict: Information about the identified employee or None if not recognized
        """
        try:
            # If there are no registered employees, return None
            if len(os.listdir(self.db_path)) == 0:
                print("No employees registered in the database")
                return {
                    'recognized': False,
                    'message': 'No employees registered in the database',
                    'status': 'empty_database'
                }
            
            # Extract face from the image first to ensure we have a face
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                align=True
            )
            
            if len(face_objs) == 0:
                return {
                    'recognized': False,
                    'message': 'No face detected in the image',
                    'status': 'no_face_detected'
                }
            
            # Store the face object for drawing purposes
            face_obj = face_objs[0]
            face_region = face_obj.get('facial_area', None)
            face_confidence = face_obj.get('confidence', None)

            # Perform face recognition against the database
            dfs = DeepFace.find(
                img_path=image,
                db_path=self.db_path,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Don't raise exception if no face detected
                align=True
            )
            
            # Check if any matches were found
            if len(dfs) > 0 and len(dfs[0]) > 0:
                best_match = dfs[0].iloc[0]
                
                # Check if the match is below the threshold (lower is better for cosine)
                if best_match['distance'] <= self.threshold:
                    # Extract employee info from the file path
                    identity_path = best_match['identity']
                    employee_id = os.path.basename(os.path.dirname(identity_path))
                    image_filename = os.path.basename(identity_path)
                    name = os.path.splitext(image_filename)[0]  # Remove file extension
                    
                    return {
                        'employee_id': employee_id,
                        'name': name,
                        'confidence': 1 - best_match['distance'],  # Convert distance to confidence
                        'recognized': True,
                        'status': 'recognized',
                        'face_region': face_region,
                        'detection_confidence': face_confidence
                    }
            
            return {
                'recognized': False,
                'message': 'Face not recognized in database',
                'status': 'not_recognized',
                'face_region': face_region,
                'detection_confidence': face_confidence
            }
            
        except Exception as e:
            print(f"Error in face identification: {e}")
            return {
                'recognized': False,
                'message': f'Error: {str(e)}',
                'status': 'error'
            }
    
    def verify_anti_spoofing(self, image):
        """Check if the provided face is real (not a photo or mask).

        Args:
            image: Either a path to an image file or a numpy array containing the image

        Returns:
            dict: Anti-spoofing check results
        """
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                align=True,
                anti_spoofing=True
            )
            
            if len(face_objs) == 0:
                return {
                    'is_real': False,
                    'message': 'No face detected',
                    'face_count': 0
                }
            
            # Check if all detected faces are real
            all_real = True
            for face_obj in face_objs:
                if not face_obj.get("is_real", False):
                    all_real = False
                    break
            
            return {
                'is_real': all_real,
                'message': 'Face verification complete',
                'face_count': len(face_objs)
            }
        
        except Exception as e:
            print(f"Error in anti-spoofing check: {e}")
            return {
                'is_real': False,
                'message': f'Error: {str(e)}',
                'face_count': 0
            }
