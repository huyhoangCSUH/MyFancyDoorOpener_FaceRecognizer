import cv2
import os
import numpy as np
from time import sleep

subjects = ["Huy Hoang", "Brad Pitt"]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()


def extract_a_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        '/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);

    # if no faces are detected then return original img
    if len(faces) == 0:
        return 0, None, None

    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return 1, gray[y:y + w, x:x + h], faces[0]

if os.path.exists("trained_faces/trained_faces.yml"):
    face_recognizer.read("trained_faces/trained_faces.yml")
    print "Trained faces loaded!"

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
        draw_text(img, label_text + " Conf: "+ str(confidence), rect[0], rect[1] - 5)

        return img, label_text
    # Start streaming webcam
    cam = cv2.VideoCapture(0)

    while 1:
        ret, frame = cam.read()
        if ret:
            have_face, face, rect = extract_a_face(frame)
            if have_face:
                frame, person = predict(frame)
            #cv2.imwrite("video/current_photo.jpg", detected_pic)
            cv2.imshow("Livestream", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        sleep(1)

            #imwrite("video/photo.jpg", img)

        #print("Predicting images...")

        # load test images
        #test_img1 = cv2.imread("test-data/test1.jpg")
        #test_img2 = cv2.imread("test-data/test2.jpg")

        # perform a prediction
        #predicted_img1, label1 = predict(test_img1)
        #predicted_img2, label2 = predict(test_img2)
        #print("Prediction complete")
        #print("test1 => " + str(label1))
        #print("Test2 => " + str(label2))

    # display both images
    #cv2.imshow(subjects[1], predicted_img1)
    #cv2.imshow(subjects[2], predicted_img2)
    #cv2.waitKey(0)
    cam.release()
    cv2.destroyAllWindows()


else:
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
                have_face, face, rect = extract_a_face(image)

                if have_face:
                    faces.append(face)
                    labels.append(label)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        return faces, labels

    print("Feeding training data...")
    faces, labels = prepare_training_data("training-data")

    face_recognizer.train(faces, np.array(labels))
    print "Finished training, saving the recognizer..."
    face_recognizer.write("trained_faces/trained_faces.yml")
