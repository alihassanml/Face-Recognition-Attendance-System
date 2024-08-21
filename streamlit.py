import streamlit as st
import cv2
import face_recognition
from datetime import datetime
import os

st.title("Face Recognition Attendance System")

# Video stream state
if 'video_started' not in st.session_state:
    st.session_state['video_started'] = False

# Loading known face encodings
fol = os.listdir('images')

def findEncodings(images):
    encodeList = []
    for img in images:
        path = 'images/' + img
        load_face = face_recognition.load_image_file(path)
        encode = face_recognition.face_encodings(load_face)[0]
        encodeList.append(encode)
    return encodeList

data = findEncodings(fol)

def label(filenames):
    labels = [filename.split('.')[0] for filename in filenames]
    return labels

name = label(fol)

def mark_attendance(name):
    now = datetime.now()
    dstring = now.strftime("%H:%M:%S %d/%m/%Y ")
    with open("Attendance.txt", "r") as file:
        lines = file.readlines()
        
    lines = [line for line in lines if not line.startswith(f"Name : {name}")]
    lines.append(f'Name : {name} Time : {dstring}\n')
    
    with open("Attendance.txt", "w") as file:
        file.writelines(lines) 

def generate_frames():
    cap = cv2.VideoCapture(0)
    face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while st.session_state['video_started']:
        ret, video = cap.read()
        if not ret:
            break
        col = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        FACES = face_cap.detectMultiScale(
            col,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in FACES:
            face_roi = video[y:y+h, x:x+w]
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            try:
                image_2_encoding = face_recognition.face_encodings(face_roi_rgb)[0]
                found_match = False
                for i in range(len(data)):
                    results = face_recognition.compare_faces([data[i]], image_2_encoding)
                    if results[0]:
                        cv2.putText(video, name[i], (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        found_match = True
                        mark_attendance(name[i])
                        break
                if not found_match:
                    cv2.putText(video, 'No Match', (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                    cv2.rectangle(video, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
            except IndexError:
                cv2.putText(video, 'Face not clear', (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        frame = cv2.imencode('.jpg', video)[1].tobytes()
        st.image(frame, channels="BGR")

    cap.release()

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Start Video"):
    st.session_state['video_started'] = True

if st.sidebar.button("Stop Video"):
    st.session_state['video_started'] = False

# Attendance display
st.sidebar.header("Attendance")
if st.sidebar.button("Show Attendance"):
    try:
        with open("Attendance.txt", "r") as file:
            attendance_data = file.read()
    except FileNotFoundError:
        attendance_data = "No attendance data available."
    st.sidebar.text(attendance_data)

# Video feed
if st.session_state['video_started']:
    generate_frames()

# Upload image form
st.sidebar.header("Upload Image")
name = st.sidebar.text_input("Enter Name")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="png")

if uploaded_file is not None and name:
    with open(f'images/{name}.png', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Image saved")
