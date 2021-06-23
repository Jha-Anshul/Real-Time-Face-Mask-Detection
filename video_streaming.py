# import all the required libraries
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

# Load model for Real time face mask detection
my_model = load_model('face_mask_classifier.h5')
print("model loaded from the disk")

# Face mask detection model summary
my_model.summary()

capture = cv2.VideoCapture(-1)
# loading haarcascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while capture.isOpened():
    ret,img = capture.read()
    # face cascade detect faces in an image and detectMultiScale return rectangle with coordinates (x,y,w,h)
    face = face_cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 4)
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        # save image for prediction
        cv2.imwrite('test.jpg', face_img)
        test_image = image.load_img('test.jpg', target_size = (200,200,3))
        test_image = image.img_to_array(test_image)
        test_image = np.array([test_image])
        prediction = my_model.predict_classes(test_image)
        # if label is 1 write no mask
        if prediction == 1:
            # draw red reactangle around face
            cv2.rectangle(img,(x,y),(x+w, y+h), (0,0,255),3)
            cv2.putText(img,'No Mask', (x+w-160, y+h-170),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
            #playsound('beep-01a.wav')
        # else write mask
        else:
            # draw green reactangle around face
            cv2.rectangle(img,(x,y),(x+w, y+h), (0,255,0),3)
            cv2.putText(img,'Mask', (x+w-160, y+h-170), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # display
    cv2.imshow('LIVE Video', img)

    # press q to quit the video
    if cv2.waitKey(1)==ord('q'):
        break

# Release the capture when everything is done
capture.release()
cv2.destroyAllWindows()
