# import cv2
# import multiprocessing
# import queue
# from mtcnn_ort import MTCNN

# # Initialize the MTCNN detector
# detector = MTCNN()

# # Process a single frame for face detection
# def process_frame(frame_queue):
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
#     out = cv2.VideoWriter('output.mp4', fourcc, 5, (1280, 720))

  
#     while True:
#         frame = frame_queue.get()
        
#         if frame is None:
#             break

#         # Detect faces using MTCNN
#         faces = detector.detect_faces(frame)

#         # Draw bounding boxes around the faces
#         for face in faces:
#             x, y, w, h = face['box']
#             confidence = face['confidence']
#             # if confidence>=0.97:
#             #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         out.write(frame)
#         # Display the processed frame
#         # cv2.imshow('RTSP Stream', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()
#     out.release()

# def capture_frames(rtsp_url, frame_queue):
#     cap = cv2.VideoCapture(rtsp_url)
#     # fps = cap.get(cv2.CAP_PROP_FPS)
#     # print(fps)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Resize the frame to reduce processing load
#         frame = cv2.resize(frame, (1280,720))

#         # Add the frame to the queue
#         if not frame_queue.full():
#             frame_queue.put(frame)

#     cap.release()

# # Main function to start both capture and processing
# def main():
#     rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"
    
#     # Create a multiprocessing queue
#     frame_queue = multiprocessing.Queue(maxsize=1)

#     # Start the frame capture process
#     capture_process = multiprocessing.Process(target=capture_frames, args=(rtsp_url, frame_queue))
#     capture_process.start()

#     # Start the face detection process
#     process_process = multiprocessing.Process(target=process_frame, args=(frame_queue,))
#     process_process.start()

#     # Wait for both processes to finish
#     capture_process.join()
#     frame_queue.put(None)  # Signal the processing process to exit
#     process_process.join()

# if __name__ == "__main__":
#     main()






import cv2
rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"
# rtsp_url = "rtsp://fake.kerberos.io/stream"
cap = cv2.VideoCapture(rtsp_url)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter('output_z.mp4', fourcc, 15, (1280, 720))
total_frames = 7*60*15
frame_cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280,720))
    cv2.imshow("win", frame)
    out.write(frame)
    
    frame_cnt +=1
    if frame_cnt >total_frames:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



