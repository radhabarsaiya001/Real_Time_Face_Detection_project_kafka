from sklearn.metrics.pairwise import cosine_similarity
from Face_Registration import extract_faces, load_dataset
from facenet_embedding_extraction import extract_embaddings
import matplotlib.pyplot as plt
import os
from numpy import expand_dims

def similarity_check(data_dir,unknown_embadding):
    files = os.listdir(data_dir)
    my_dict = {}
    lst =[]
    for i in files:
        file_path = os.path.join(data_dir,i)
        face_arr = extract_faces(file_path)
        face_embadding = expand_dims(extract_embaddings(face_arr),axis=0)
        file_name = os.path.basename(file_path).split('.')[0]
        similarity_score =cosine_similarity(face_embadding,unknown_embadding)
        my_dict[f"{file_name}"] = f"{similarity_score}"
    lst.append(my_dict)
    return lst


# data_dir = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\Data"

unkown_img = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\testing_data\radha.jpeg"
unkown_img_arr= extract_faces(unkown_img)
unknown_embadding = expand_dims(extract_embaddings(unkown_img_arr),axis=0)
# print(similarity_check(data_dir,unknown_embadding))




file_path = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\testing_data\radha_an.jpeg"
face_arr = extract_faces(file_path)
face_embadding = expand_dims(extract_embaddings(face_arr),axis=0)
similarity_score =cosine_similarity(face_embadding,unknown_embadding)
print(f"{similarity_score[0][0]*100} %")




















# similarity_score = cosine_similarity(face_embadding_1,face_embadding_2)
# print(similarity_score)


# print(face_embadding_1)

