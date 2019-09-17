from matplotlib import pyplot as plt
from facemodel import face_recognition
import cv2
from mtcnn.mtcnn import MTCNN

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
detector = MTCNN()

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Use MTCNN to detect faces
        result = detector.detect_faces(rgb)

        if result != []:
            for face in result:
                bounding_box = face['box']
                # keypoints = face['keypoints']
                x, y, w, h = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
                rect_face = cv2.rectangle(frame, (x, y), (x+w, y+h), (46, 204, 113), 2)
                face = rgb[y:y+h, x:x+w]

                predicted_name, class_probability = face_recognition(face)

                print("Result: ", predicted_name, class_probability)

                if class_probability >= 50:
                    rect_face = cv2.rectangle(frame, (x, y-15), (x+w, y+10), (46, 204, 113), -1)
                    cv2.putText(rect_face, predicted_name, (x+1, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 236, 240, 241), 2)
                else:
                    rect_face = cv2.rectangle(frame, (x, y-15), (x+w, y+10), (46, 204, 113), -1)
                    cv2.putText(rect_face, "Unknown", (x+1, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 236, 240, 241), 2)

                # cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                # cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                # cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
                # cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                # cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()