# Face ID Attendance System

A powerful facial recognition-based attendance system that uses DeepFace for employee identification. The system records check-in and check-out times, tracks late arrivals, and generates attendance reports.

## Features

- **Face Recognition**: Identifies employees using state-of-the-art facial recognition
- **Check-in/Check-out**: Records when employees arrive and leave
- **Late Detection**: Automatically marks employees who arrive late
- **Anti-Spoofing**: Detects fake faces to prevent attendance fraud
- **Reports**: Generates daily and monthly attendance reports
- **User-friendly Interface**: Simple camera-based interface for employees

## Project Structure

```
FaceIDSystem/
├── main.py                    # Main application entry point
├── config.py                  # Configuration settings
├── face_recognition_module.py # Face recognition functionality
├── attendance_manager.py      # Attendance record management
├── ui_manager.py              # User interface handling
├── requirements.txt           # Python dependencies
├── employees/                 # Employee face database
└── attendance_records/        # CSV files of attendance data
```

## Installation

1. Install the required dependencies:

```shell
pip install -r requirements.txt
```

2. Ensure you have a working webcam.

## Usage

### Registering an Employee

To register a new employee in the system:

```shell
python main.py register --id EMP001 --name "John Doe" --image "path/to/photo.jpg"
```

### Starting the Attendance System

To start the interactive attendance system:

```shell
python main.py start
```

Once the system is running:
- Press 'M' to toggle between Check-In and Check-Out modes
- Press 'ESC' to exit the system

### Generating Reports

To generate a monthly attendance report:

```shell
python main.py report --year 2025 --month 4
```

If year and month are not specified, the current month's report will be generated.

## Customization

You can customize the system by modifying the settings in `config.py`:

- Working hours and late thresholds
- Face recognition model and detector
- Camera settings
- UI appearance

## How It Works

1. **Face Detection**: The system detects faces in the webcam feed
2. **Anti-Spoofing**: Ensures the detected face is real, not a photo or mask
3. **Recognition**: Compares the face against the employee database
4. **Attendance Recording**: Records time and status in CSV files
5. **Reporting**: Generates reports based on the recorded data

## Dependencies

- deepface: Facial recognition framework
- opencv-python: Computer vision library for camera handling
- pandas: Data manipulation for attendance records
- numpy: Numerical computing
- matplotlib: Visualization (for potential future features)
