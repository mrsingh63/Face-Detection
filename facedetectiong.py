import cv2
from random import randrange as r

trainedData=cv2.CascadeClassifier('face.xml')

#choose image
img=cv2.imread('group.jpg')



greyImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faceCoordinates=trainedData.detectMultiScale(greyImage)

for x,y,w,h in faceCoordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)

# show/display image
cv2.imshow('Single Person', img)

cv2.waitKey()


print('End of program')