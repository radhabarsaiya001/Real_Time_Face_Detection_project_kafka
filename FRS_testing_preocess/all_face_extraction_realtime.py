# # from mtcnn_ort import MTCNN
# # import cv2
# # import time
# # import os 
# # from numpy import asarray, expand_dims, load
# # from numpy import savez_compressed
# # import matplotlib.pyplot as plt
# # from sklearn.metrics.pairwise import cosine_similarity
# # import joblib
# # from keras_facenet import FaceNet
# # from multiprocessing import Process, Queue
# # svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\optimized_face_model.pkl")
# # label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\optimized_label_encoder.pkl")
# # model = FaceNet()
# # detector = MTCNN()
# # data = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\face_embeddings_wth_labels.npz")


# # def extract_multiple_faces(frame_queue, frame_path, required_size=(160, 160)):
# #     img = cv2.imread(frame_path)
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #for higher accuracy of MTCNN model
# #     face_pstn_lst = detector.detect_faces(img)
# #     frame_face_arr = []
# #     for i in range(len(face_pstn_lst)):
# #         x,y,w,h = face_pstn_lst[i]['box']
# #         face_arr = img[y:y+h, x:x+w]
# #         face_arr = cv2.resize(face_arr,required_size)
# #         face_arr = asarray(face_arr)
# #         frame_face_arr.append(face_arr)
# #     frame_face_arr = asarray(frame_face_arr)
# #     # return frame_face_arr
# #     frame_queue.put(frame_face_arr)

# # def extract_multiple_embaddings(frame_queue,embedding_queue):
# #     frame_face_arr = frame_queue.get()
# #     frame_all_embedding = []
# #     for i in frame_face_arr:
# #         samples = expand_dims(i,axis=0)  
# #         yhat = model.embeddings(samples)
# #         frame_all_embedding.append(yhat[0])

# #     frame_all_embedding = asarray(frame_all_embedding)
# #     # return frame_all_embedding
# #     embedding_queue.put(frame_all_embedding)
# # def model_testing_on_realtime_sample(embedding_queue, model):
# #     frame_all_embedding = embedding_queue.get()
# #     for i in frame_all_embedding:
# #         sample_embedding = expand_dims(i, axis=0)
# #         train_y_pred = model.predict(sample_embedding)
# #         predicted_class = label_encoder.inverse_transform(train_y_pred)
# #         cnt = 0
# #         for i in data['arr_1']:
# #             if i == predicted_class[0]:
# #                 training_embedding = data['arr_0'][cnt][:]
# #                 training_embedding = expand_dims(training_embedding, axis=0)  
# #                 break
# #             cnt+=1
# #         similarity_score = cosine_similarity(training_embedding,sample_embedding)
# #         similarity_score = round(similarity_score[0][0] * 100,3)
# #         if similarity_score < 60 :
# #             predicted_class = 'Unknown'
# #             print(predicted_class)  
# #             yield predicted_class, f"{similarity_score} %"
# #         else:
# #             print(predicted_class)
# #             yield predicted_class[0], f"{similarity_score} %"
    

# # # frame_path = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\testing_data\multiple_face_testing.jpeg"
# # frame_path = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\testing_data\radha_an.jpeg"
# # # # frame_face_arr = extract_multiple_faces(frame_path)
# # # frame_all_embedding = extract_multiple_embaddings(frame_face_arr)
# # # model_testing =model_testing_on_realtime_sample(svc_model, frame_all_embedding)
# # # for i,j in model_testing:
# # #     print(i, j)


# # if __name__=="__main__":
# #     frame_queue = Queue() 
# #     embedding_queue = Queue()
# #     frame_face_arr = Process(target=extract_multiple_faces, args=(frame_queue, frame_path)) 
# #     embedding = Process(target=extract_multiple_embaddings, args=(frame_queue,embedding_queue)) 
# #     model_testing = Process(target=model_testing_on_realtime_sample, args=(embedding_queue,model)) 

# #     frame_face_arr.start()
# #     embedding.start()
# #     model_testing.start()

# #     frame_face_arr.join()
# #     embedding.join()
# #     model_testing.join()





import cv2
import multiprocessing
from mtcnn_ort import MTCNN
from numpy import asarray, expand_dims
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from keras_facenet import FaceNet
from numpy import asarray, expand_dims, load

# Load pre-trained models
svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_face_model.pkl")
label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_label_encoder.pkl")
model = FaceNet()
detector = MTCNN()
data = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\face_embeddings_wth_labels.npz")


# Process a single frame for face detection
def detect_faces(frame_queue, embedding_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)
        frame_face_arr = []

        # Process each face detected in the frame
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_arr = frame[y:y + h, x:x + w]
            face_arr = cv2.resize(face_arr, (160, 160))
            frame_face_arr.append(face_arr)
       
        if len(frame_face_arr) > 0:
            frame_face_arr = asarray(frame_face_arr)
            embedding_queue.put(frame_face_arr)


# Capture frames from the RTSP stream
def capture_frames(rtsp_url, frame_queue):
    cap = cv2.VideoCapture(rtsp_url)
    # cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

# Extract face embeddings using FaceNet
def extract_embeddings(embedding_queue, recognition_queue):
    while True:
        frame_face_arr = embedding_queue.get()
        if frame_face_arr is None:
            break
        frame_all_embedding = []
        for face in frame_face_arr:
            samples = expand_dims(face, axis=0)
            yhat = model.embeddings(samples)
            frame_all_embedding.append(yhat[0])
        if len(frame_all_embedding) > 0:
            recognition_queue.put(asarray(frame_all_embedding))

# Perform classification and similarity checks on embeddings
def recognize_faces(recognition_queue):
    while True:
        frame_all_embedding = recognition_queue.get()
        if frame_all_embedding is None:
            break

        for embedding in frame_all_embedding:
            sample_embedding = expand_dims(embedding, axis=0)
            predicted_class = svc_model.predict(sample_embedding)
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]
            cnt = 0
            for i in data['arr_1']:
                if i == predicted_label:
                    training_embedding = data['arr_0'][cnt][:]
                    training_embedding = expand_dims(training_embedding, axis=0)
                    break
                cnt += 1
            similarity_score = cosine_similarity(training_embedding, sample_embedding)
            similarity_score = round(similarity_score[0][0] * 100, 3)
            if similarity_score < 60:
                print('Unknown', f"{similarity_score}%")
            else:
                print(predicted_label, f"{similarity_score}%")

# Main function to start the processes
def main():
    rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"

    # Create multiprocessing queues
    frame_queue = multiprocessing.Queue(maxsize=5)
    embedding_queue = multiprocessing.Queue(maxsize=5)
    recognition_queue = multiprocessing.Queue(maxsize=5)

    # Start the frame capture process
    capture_process = multiprocessing.Process(target=capture_frames, args=(rtsp_url, frame_queue))
    capture_process.start()

    # Start the face detection process
    detection_process = multiprocessing.Process(target=detect_faces, args=(frame_queue, embedding_queue))
    detection_process.start()

    # Start the face embedding extraction process
    embedding_process = multiprocessing.Process(target=extract_embeddings, args=(embedding_queue, recognition_queue))
    embedding_process.start()

    # Start the face recognition process
    recognition_process = multiprocessing.Process(target=recognize_faces, args=(recognition_queue,))
    recognition_process.start()

    # Wait for the processes to finish
    capture_process.join()
    frame_queue.put(None)  # Signal the processes to exit
    detection_process.join()
    embedding_queue.put(None)  # Signal the embedding process to exit
    embedding_process.join()
    recognition_queue.put(None)  # Signal the recognition process to exit
    recognition_process.join()

if __name__ == "__main__":
    main()




# from mtcnn_ort import MTCNN
# import cv2
# import time
# import os 
# from numpy import asarray, expand_dims, load
# from numpy import savez_compressed
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from keras_facenet import FaceNet
# svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\optimized_face_model.pkl")
# label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\optimized_label_encoder.pkl")
# model = FaceNet()
# detector = MTCNN()
# data = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\face_embeddings_wth_labels.npz")


# def extract_multiple_faces(frame_path, required_size=(160, 160)):
#     img = cv2.imread(frame_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      #for higher accuracy of MTCNN model
#     face_pstn_lst = detector.detect_faces(img)
#     frame_face_arr = []
#     for i in range(len(face_pstn_lst)):
#         x,y,w,h = face_pstn_lst[i]['box']
#         face_arr = img[y:y+h, x:x+w]
#         face_arr = cv2.resize(face_arr,required_size)
#         face_arr = asarray(face_arr)
#         frame_face_arr.append(face_arr)
#     frame_face_arr = asarray(frame_face_arr)
#     return frame_face_arr

# def extract_multiple_embaddings(frame_face_arr):
#     frame_all_embedding = []
#     for i in frame_face_arr:
#         samples = expand_dims(i,axis=0)  
#         yhat = model.embeddings(samples)
#         frame_all_embedding.append(yhat[0])

#     frame_all_embedding = asarray(frame_all_embedding)
#     return frame_all_embedding

# def model_testing_on_realtime_sample(model, frame_all_embedding):
#     for i in frame_all_embedding:
#         sample_embedding = expand_dims(i, axis=0)
#         train_y_pred = model.predict(sample_embedding)
#         predicted_class = label_encoder.inverse_transform(train_y_pred)
#         cnt = 0
#         for i in data['arr_1']:
#             if i == predicted_class[0]:
#                 training_embedding = data['arr_0'][cnt][:]
#                 training_embedding = expand_dims(training_embedding, axis=0)  
#                 break
#             cnt+=1
#         similarity_score = cosine_similarity(training_embedding,sample_embedding)
#         similarity_score = round(similarity_score[0][0] * 100,3)
#         if similarity_score < 60 :
#             predicted_class = 'Unknown'
#             yield predicted_class, f"{similarity_score} %"
#         else:
#             yield predicted_class[0], f"{similarity_score} %"
    

# frame_path = r"C:\Users\hp\Downloads\WhatsApp Image 2024-09-27 at 18.01.59.jpeg"
# #frame_path = r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\testing_data\radha_an.jpeg"
# frame_face_arr = extract_multiple_faces(frame_path)
# frame_all_embedding = extract_multiple_embaddings(frame_face_arr)
# model_testing =model_testing_on_realtime_sample(svc_model, frame_all_embedding)
# for i,j in model_testing:
#   print(i, j)
