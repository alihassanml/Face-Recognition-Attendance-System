# Face Recognition Attandance System

This project uses FastAPI to deploy an attendance system integrated with face recognition. It captures live video from a webcam, detects faces, matches them with a pre-defined list of known faces, and marks attendance accordingly. 

## Features

- **Live Face Recognition**: Captures video and recognizes faces in real-time.
- **Attendance Marking**: Logs attendance with timestamps when a face is recognized.
- **Image Upload**: Allows uploading new face images to be added to the system.
- **Web Interface**: Provides a web interface for viewing attendance and controlling video feed.

## Requirements

- Python 3.8+
- FastAPI
- OpenCV
- `face_recognition` library
- `Jinja2` for templating

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alihassanml/Face-Recognition-Attandance-System.git
   cd Attandance-System-Face-Recognition
   ```

2. Install the dependencies:
   ```bash
   pip install fastapi[all] opencv-python face_recognition
   ```

## Usage

1. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

2. **Navigate to the web interface**:
   Open your browser and go to `http://localhost:8000/`.

3. **Upload Images**:
   Use the `/upload_image` endpoint to upload face images. 

4. **Start/Stop Video Feed**:
   - Start video feed: `http://localhost:8000/start_video`
   - Stop video feed: `http://localhost:8000/stop_video`

5. **View Attendance**:
   The attendance records can be viewed at the root endpoint: `http://localhost:8000/`.

## Code Structure

- `main.py`: Contains the FastAPI application and the core logic for face recognition and attendance marking.
- `templates/`: Directory for HTML templates.
- `static/`: Directory for static files like CSS and JavaScript.
- `images/`: Directory for storing uploaded face images.
- `Attendance.txt`: File where attendance records are saved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
