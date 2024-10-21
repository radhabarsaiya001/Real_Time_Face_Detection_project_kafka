from mtcnn_ort import MTCNN
import cv2
import time

# initialize the MTCNN detector
detector = MTCNN()
# video_path = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"
cap = cv2.VideoCapture(0)
while True:
    try:
        # read the frame from the camera
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (640, 480))
        # detect faces using MTCNN
        faces = detector.detect_faces(frame)
        print(faces)
        time.sleep(3)

        # draw bounding boxes around the faces
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # show the resulting frame
        cv2.imshow('RTSP Stream', frame)

        # press 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()