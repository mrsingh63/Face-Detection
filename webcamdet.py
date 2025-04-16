import cv2
from random import randrange as r

trainedData=cv2.CascadeClassifier('face.xml')

#choose image
webcam = cv2.VideoCapture(0)
while True:
    success,frame=webcam.read()

    greyImage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faceCoordinates=trainedData.detectMultiScale(greyImage)

    for x,y,w,h in faceCoordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)

    # show/display image
    cv2.imshow('Single Person', frame)

    key=cv2.waitKey(1)
    if(key==82 or key==113):
        break

webcam.release

print('End of program')