import joblib
from numpy import load
from numpy import expand_dims
from facenet_embedding_extraction import extract_embaddings
from sklearn.preprocessing import LabelEncoder, Normalizer
from Face_Registration import extract_faces
from sklearn.metrics.pairwise import cosine_similarity

svc_model = joblib.load('optimized_face_model.pkl')
label_encoder = joblib.load('optimized_label_encoder.pkl')
data = load('face_embeddings_wth_labels.npz')
cnt = 0

def model_testing_on_realtime_sample(model, frame_path):
    face_arr = extract_faces(frame_path)
    face_embadding = extract_embaddings(face_arr)
    sample_embaddeing = expand_dims(face_embadding, axis=0)
    train_y_pred = model.predict(sample_embaddeing)
    predicted_class = label_encoder.inverse_transform(train_y_pred)
    return sample_embaddeing, predicted_class

def get_training_embaddeing(model, frame_path):
    global cnt
    sample_embedding, predicting_class = model_testing_on_realtime_sample(model, frame_path)
    for i in data['arr_1']:
        if i == predicting_class[0]:
            training_embedding = data['arr_0'][cnt][:]
            training_embedding = expand_dims(training_embedding, axis=0)
            break
        cnt+=1
    return sample_embedding, training_embedding, predicting_class

def similarity_check(model, frame_path):
    sample_embedding , training_embedding, predicted_class =  get_training_embaddeing(model, frame_path)
    similarity_score = cosine_similarity(training_embedding,sample_embedding)
    similarity_score = round(similarity_score[0][0] * 100,3)
    return predicted_class[0], f"{similarity_score} %"

frame_path = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\testing_data\sakshi.jpeg"
print(similarity_check(svc_model, frame_path))


# training_embedding = data['arr_0'][cnt][:]
# print(training_embedding)


# similarity check function