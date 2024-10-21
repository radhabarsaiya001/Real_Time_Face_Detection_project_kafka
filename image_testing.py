
from mtcnn_ort import MTCNN
import cv2
import time
import os 
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
detector = MTCNN()
frame_path = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\Data\nikki.jpeg"
def extract_multiple_faces(frame_path):
    img = cv2.imread(frame_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #for higher accuracy of MTCNN model
    face_pstn_lst = detector.detect_faces(img)
    print(face_pstn_lst)
    for i in face_pstn_lst:
        x,y,w,h = face_pstn_lst[0]['box']
        # print("if face detected")
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('RTSP Stream', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
extract_multiple_faces(frame_path)

        # face_arr = img[y:y+h, x:x+w]
        # face_arr = cv2.resize(face_arr,required_size)
        # face_arr = asarray(face_arr)
        # frame_face_arr.append(face_arr)
    # frame_face_arr = asarray(frame_face_arr)
    # return frame_face_arr