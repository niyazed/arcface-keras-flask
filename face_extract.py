
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

cap = cv2.VideoCapture(0)


while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # rgb = frame
    #Use MTCNN to detect faces
    result = detector.detect_faces(rgb)

    if result != []:
        for face in result:
            bounding_box = face['box']
            # keypoints = face['keypoints']
            x, y, w, h = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
            rect_face = cv2.rectangle(frame, (x, y), (x+w, y+h), (46, 204, 113), 2)
            face = rgb[y:y+h, x:x+w]

            if face.shape[0]*face.shape[1] > 0:
                cv2.imwrite('extracted/'+ str(face.shape[0]*face.shape[1]) + '.jpg', face)


    #display resulting frame
    cv2.imshow('frame',frame)
    # out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #When everything's done, release capture

cap.release()
cv2.destroyAllWindows()

