from fastapi import FastAPI, Request, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import face_recognition
from datetime import datetime
import os
from fastapi.staticfiles import StaticFiles
from typing import Optional

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

cap = cv2.VideoCapture(0)
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    while True:
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

        ret, buffer = cv2.imencode('.jpg', video)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        with open("Attendance.txt", "r") as file:
            lines = file.readlines()
            attendance_data = "<br>".join(line.strip() for line in lines)
    except FileNotFoundError:
        attendance_data = "No attendance data available."
    return templates.TemplateResponse("index.html", {"request": request, "attendance_data": attendance_data})


@app.post("/upload_image")
async def upload_image(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    # Save the uploaded file
    path = f'images/{name}.png'
    with open(path, 'wb') as buffer:
        buffer.write(await file.read())
    return {"message": "Image saved", "path": path}


@app.get("/start_video")
def start_video():
    global cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return {"message": "Video started"}

@app.get("/stop_video")
def stop_video():
    global cap
    if cap.isOpened():
        cap.release()
    return {"message": "Video stopped"}
