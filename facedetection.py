import cv2

trainedData=cv2.CascadeClassifier('face.xml')

#choose image
img=cv2.imread('1691926293.jpeg')



greyImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faceCoordinates=trainedData.detectMultiScale(greyImage)
x,y,w,h=faceCoordinates[0]

cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# show/display image
cv2.imshow('Single Person', img)

cv2.waitKey()


print('End of program')