import cv2
import time
import os 
from numpy import asarray, load
from numpy import expand_dims
from numpy import savez_compressed
from Face_Registration import extract_faces
from keras_facenet import FaceNet
import  matplotlib.pyplot as plt
model = FaceNet()

def extract_embaddings(frame_arr):
    samples = expand_dims(frame_arr,axis=0)  
    yhat = model.embeddings(samples)
    return yhat[0]

def all_embeddings(training_data_file_path):
    data = load(training_data_file_path)    #npz file of face arrays
    face_embedding = []
    face_labels=[]
    for i in range(len(data['arr_0'])):
        # plt.imshow(data['arr_0'][i])
        # plt.show()
        # print(data['arr_1'][i])
        lx = extract_embaddings(data['arr_0'][i])
        face_embedding.append(lx)
        face_label = data['arr_1'][i]
        face_labels.append(face_label)
    store_face_embeddings(face_embedding,face_labels)

def store_face_embeddings(face_embedding, face_labels):
    cwd = os.getcwd()
    folder_path = "weight_model_files"
    path = os.path.join(cwd, folder_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    faces_embedding_file_name = "face_embeddings_wth_labels.npz"
    file_path = os.path.join(path, faces_embedding_file_name)
    savez_compressed(file_path,face_embedding, face_labels)


# training_data_file_path = r'C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\training_data.npz'
# all_embeddings(training_data_file_path)
