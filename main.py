import cv2
import os
import numpy as np
from time import sleep
import socket
import sys
import time


subjects = ["Huy Hoang", "Brad Pitt", "Tahsin", "Yong Li"]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()


def extract_a_face(img, scalefact):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        '/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scalefact, minNeighbors=5, minSize=(100, 100));

    # if no faces are detected then return original img
    if len(faces) == 0:
        return 0, None, None

    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return 1, gray[y:y + w, x:x + h], faces[0]


if not os.path.exists("trained_faces/trained_faces.yml"):
    print "No trained faces detected!"

    def prepare_training_data(data_folder_path):

        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []

        for dir_name in dirs:
            if not dir_name.startswith("person"):
                continue
            label = int(dir_name.replace("person", ""))
            subject_dir_path = data_folder_path + "/" + dir_name

            # get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:

                # ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue

                # build image path
                # sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                # read image
                image = cv2.imread(image_path)

                # detect face
                have_face, face, rect = extract_a_face(image, 1.05)

                if have_face:
                    faces.append(face)
                    labels.append(label)

        return faces, labels

    print("Feeding training data...")
    faces, labels = prepare_training_data("training-data")

    face_recognizer.train(faces, np.array(labels))
    print "Finished training, saving the recognizer..."
    face_recognizer.write("trained_faces/trained_faces.yml")


face_recognizer.read("trained_faces/trained_faces.yml")
print "Trained data loaded!"


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(img):
    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text + " Conf: "+ str(int(confidence)), rect[0], rect[1] - 5)
    return img, label_text


HOST = 'huyhoang.io'
PORT = 9999


def send_video_file(file_to_send):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    s.send("photo")
    print "pilot sent"
    s.close()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    file_to_send = open(file_to_send, 'rb')
    l = file_to_send.read(1024)
    while l:
        #print 'Sending...'
        s.send(l)
        l = file_to_send.read(1024)
    # s.shutdown(socket.SHUT_WR)
    s.close()
    file_to_send.close()

def send_auth_stat(message):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.send("auth_stat")
    print "auth pilot sent!"
    s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.send(message)
    s.close()


# Start streaming webcam
cam = cv2.VideoCapture(0)
while 1:
    ret, frame = cam.read()
    if ret:
        have_face, face, rect = extract_a_face(frame, 1.2)
        person_info = ""
        if have_face:
            frame, person = predict(frame)
            person_info = "{person: " + person + ", distance: 500mm }"
        else:
            person_info = "{person: none, distance: 0mm}"
        #cv2.imshow("Livestream", frame)
    frame = cv2.resize(frame, (640, 360))
    cv2.imwrite("webcam_cap.jpg", frame)

    print "Sending a frame"

    send_video_file("webcam_cap.jpg")
    send_auth_stat(person_info)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    time.sleep(2)

cam.release()
cv2.destroyAllWindows()



