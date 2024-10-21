import cv2
import threading
import queue
import time
from mtcnn import MTCNN

# Initialize the MTCNN detector
detector = MTCNN()

# Buffer to hold the frames (queue for communication between threads)
frame_queue = queue.Queue()

# Thread for capturing frames from RTSP stream
def capture_frames(rtsp_url, frame_queue):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the frame to reduce processing load
        frame = cv2.resize(frame, (640, 480))

        # Add the frame to the queue if there's space
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            print("Queue is full, dropping frame")

    cap.release()

# Thread for processing frames (Face Detection)
def process_frames(frame_queue):
    while True:
        if not frame_queue.empty():
            # Get the frame from the queue
            frame = frame_queue.get()

            # Detect faces using MTCNN
            faces = detector.detect_faces(frame)

            # Draw bounding boxes around the faces
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the processed frame
            cv2.imshow('RTSP Stream', frame)

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Allow other threads to run
        time.sleep(0.01)

    cv2.destroyAllWindows()

# Main function to start both threads
def main():
    rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"

    # Create and start the frame capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, frame_queue))
    capture_thread.daemon = True
    capture_thread.start()

    # Create and start the face detection thread
    process_thread = threading.Thread(target=process_frames, args=(frame_queue,))
    process_thread.daemon = True
    process_thread.start()

    # Keep the main thread alive until 'q' is pressed
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")

if __name__ == "__main__":
    main()
