# import cv2
# import multiprocessing
# import queue
# from mtcnn_ort import MTCNN

# # Initialize the MTCNN detector
# detector = MTCNN()

# # Process a single frame for face detection
# def process_frame(frame_queue):
#     while True:
#         frame = frame_queue.get()
        
#         if frame is None:
#             break

#         # Detect faces using MTCNN
#         faces = detector.detect_faces(frame)
#         print(faces)
#         # Draw bounding boxes around the faces
#         for face in faces:
#             x, y, w, h = face['box']
#             confidence = face['confidence']
#             if confidence>=0.99:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Display the processed frame
#         cv2.imshow('RTSP Stream', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()

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
#     # # Create a multiprocessing queue
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
import multiprocessing
import os
import queue
from mtcnn_ort import MTCNN

# Initialize the MTCNN detector
detector = MTCNN()

# Directory to save face frames
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

# Process a single frame for face detection
def process_frame(frame_queue):
    face_count = 0  # Counter to name saved face frames

    while True:
        frame = frame_queue.get()

        if frame is None:
            break

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)
        print(faces)

        # Draw bounding boxes around the faces and save face frames
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            if confidence >= 0.99:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract face from the frame
                face_frame = frame[y:y + h, x:x + w]

                # Save the face frame as an image
                face_filename = os.path.join(output_dir, f"face_{face_count}.jpg")
                cv2.imwrite(face_filename, face_frame)
                print(f"Saved face frame as {face_filename}")
                face_count += 1

        # Display the processed frame with bounding boxes
        cv2.imshow('RTSP Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def capture_frames(rtsp_url, frame_queue):
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the frame to reduce processing load
        frame = cv2.resize(frame, (1280, 720))

        # Add the frame to the queue
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()

# Main function to start both capture and processing
def main():
    rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"
    # Create a multiprocessing queue
    frame_queue = multiprocessing.Queue(maxsize=1)

    # Start the frame capture process
    capture_process = multiprocessing.Process(target=capture_frames, args=(rtsp_url, frame_queue))
    capture_process.start()

    # Start the face detection process
    process_process = multiprocessing.Process(target=process_frame, args=(frame_queue,))
    process_process.start()

    # Wait for both processes to finish
    capture_process.join()
    frame_queue.put(None)  # Signal the processing process to exit
    process_process.join()

if __name__ == "__main__":
    main()
