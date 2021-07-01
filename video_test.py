import cv2
import keras.models
import numpy as np

label = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']

faceDetected = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
emotionModel = keras.models.load_model("outputs\models\mixVX_48x48_fer2013plus_32bs_43-0.83.hdf5")


def getFaceCrop(model, RGB_image):
    gray_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray_image, 1.3, 5)

    for face_coordinates in faces:
        x, y, width, height = face_coordinates
        x1, x2, y1, y2 = x, x + width, y, y + height
        gray_face = gray_image[y1:y2, x1:x2]
        return 1, gray_face, [(x1, y1), (x2, y2)]
    return -1, None, None


cap = cv2.VideoCapture(0)
assert cap.isOpened(), "摄像头未打开"
while True:
    ret, frame = cap.read()
    if ret:
        sign, gray, pos = getFaceCrop(faceDetected, frame)
        if sign == 1:
            res = emotionModel.predict(
                np.expand_dims(np.expand_dims(np.array(cv2.resize(gray, (48, 48))).astype("float32"), axis=0), axis=-1))
            emotion = label[np.argmax(res)]

            cv2.rectangle(frame, pos[0], pos[1], (0, 255, 0), 1)
            cv2.putText(frame, emotion, pos[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("", frame)
        key = cv2.waitKey(1)
        if key == "27":
            break
