from matplotlib import pyplot as plt
from facemodel import face_recognition
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
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

    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break #When everything's done, release capture

cap.release()
cv2.destroyAllWindows()
