from matplotlib import pyplot as plt
from facemodel import face_recognition
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(rgb, 1.3, 5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:

            rect_face = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = rgb[y:y+h, x:x+w]

            predict_name, class_probability = face_recognition(face)

            if class_probability >= 50:
                cv2.putText(rect_face, predict_name, (x+1, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(rect_face, "Unknown", (x+1, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()