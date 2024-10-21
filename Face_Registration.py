# *************************training_on_large_dataset_at_a_time*************************

from mtcnn_ort import MTCNN
import cv2
import time
import os 
from numpy import asarray
from numpy import savez_compressed
import matplotlib.pyplot as plt
detector = MTCNN()

def extract_faces(frame_path, required_size=(160, 160)):
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #for higher accuracy of MTCNN model
    face_pstn_lst = detector.detect_faces(img)
    x,y,w,h = face_pstn_lst[0]['box']
    face_arr = img[y:y+h, x:x+w]
    face_arr = cv2.resize(face_arr,required_size)
    face_arr = asarray(face_arr)
    return face_arr

def load_dataset(data_dir):
    faces = []
    labels = []
    if os.path.isdir(data_dir):
        lst = os.listdir(data_dir)    
    for i in lst:
        frame_path = os.path.join(data_dir, i)
        face = extract_faces(frame_path=frame_path)
        # plt.imshow(face)
        # plt.show()
        # time.sleep(2)
        faces.append(face)

        frame_name = os.path.basename(frame_path)
        label = frame_name.split('.')[0]
        # print(label)
        labels.append(label)    
    return asarray(faces), asarray(labels)

def training_data(data_dir,folder_path):
    faces, labels = load_dataset(data_dir=data_dir)
    cwd = os.getcwd()
    path = os.path.join(cwd, folder_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    faces_array_file_name = "training_data.npz"
    file_path = os.path.join(path, faces_array_file_name)
    savez_compressed(file_path,faces,labels)

data_dir= r'C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\Data'
folder_path = "weight_model_files"

####### **************this function write the array of face image in compressed file.
# training_data(data_dir,folder_path)


