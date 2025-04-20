def process_frame(frame, system_state, admin_password="", employee_name="", employee_id=""):
    if frame is None or not system_state.camera_active:
        # Return a blank frame with message if camera is off
        blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
        cv2.putText(blank_frame, "Camera Off", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return blank_frame, "Camera Off", "Press 'Turn On Camera' to start"
    
    employees = attendance_manager.load_employee_database()
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