import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
from pygame import mixer


prev_frame_time = 0
new_frame_time = 0

# load model
model = load_model('models/model-100.h5')

# load beep sound
mixer.init()
sound = mixer.Sound('beep.wav')
sound.set_volume(1)

# text font
font = cv2.FONT_HERSHEY_SIMPLEX 

# frame width, frame height
width = 640
height = 480

# input image size
image_width = 128
image_height = 128

# choose webcam source 0, 1, 2,...
cap = cv2.VideoCapture(1)
# set width, height of frame
cap.set(3, width)
cap.set(4, height)

# load eye cascade
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

# threshold and count value
threshold = 0.7
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    # flip the frame
    frame = cv2.flip(frame, 1)

    # draw black rectangle on frame
    cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (0, height-50), (width, height), (0, 0, 0), thickness=-1)

    # convert color frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # show fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, "FPS: " + fps, (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # detect eye
    eye_rect = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (ex, ey, ew, eh) in eye_rect:
        roi_eye = gray[ey:ey+eh, ex:ex+ew]
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), thickness=1)
        
        # preprocessing images
        eye_images = cv2.resize(roi_eye, (image_width, image_height))
        x = img_to_array(eye_images)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        images = np.vstack([x]) 
        # predict images
        val = model.predict(images)
        if val > threshold:
            count = count - 10
            eye_status = 'Eyes Opened'
            if count < 0:
                count = 0
        else:
            count = count + 1
            eye_status = 'Eyes Closed'     
            if count > 15:
                sound.play()
                # show alert!!!
                cv2.rectangle(frame, (4, 54), (width-4, height-54), (0, 0, 255), thickness=4)
                cv2.putText(frame, 'ALERT!!!', (350, 150), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                sound.stop()
        print(eye_status)

        # draw text in frame
        cv2.rectangle(frame, (0, height-50), (width-300, height), (255, 0, 0), thickness=-1)
        cv2.putText(frame, eye_status, (50, 460), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(count), (300, 460), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
