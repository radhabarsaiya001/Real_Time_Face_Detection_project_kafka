from numpy import load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
from Face_Registration import extract_faces
from facenet_embedding_extraction import extract_embaddings
from numpy import expand_dims
import os

def recognizing_model(face_embedding_weight_file):
    data = load(face_embedding_weight_file)
    train_x = data['arr_0']
    train_y = data['arr_1']
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)

    # # Create the SVC model

    svc_model = SVC(kernel='linear', probability=True)
    svc_model.fit(train_x, train_y)
    model_pickel_files(svc_model, label_encoder)

def model_pickel_files(svc_model, label_encoder):
    cwd = os.getcwd()
    folder_path = "model_pickel_files"
    path = os.path.join(cwd, folder_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    optimized_face_model_filename = "optimized_face_model.pkl"
    optimized_face_model_file_path = os.path.join(path, optimized_face_model_filename)
    optimized_label_encoder_filename = "optimized_label_encoder.pkl"
    optimized_label_encoder_file_path = os.path.join(path, optimized_label_encoder_filename)
    # Save the optimized model and label encoder

    # joblib.dump(svc_model, optimized_face_model_file_path)
    # joblib.dump(label_encoder, optimized_label_encoder_file_path)




# face_embedding_weight_file = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\face_embeddings_wth_labels.npz"
# recognizing_model(face_embedding_weight_file)