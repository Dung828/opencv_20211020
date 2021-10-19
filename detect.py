import cv2
import numpy as np

# import cascade file for facial recognition
#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# if you want to detect any object for example eyes, use one more layer of classifier as below:
#eyeCascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml')


width=640
height=480
cap = cv2.VideoCapture(0)
#cap.set(3,width) # set Width
#cap.set(4,height) # set Height

while True:
    ret, frame = cap.read ()
#    print('ret',ret)
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    x=0;y=0;w=0;h=0;
    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
 
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#        print(x,y,w,h)
        x1=x
        y1=y
        x2=x+w
        y2=y+h
        ROI = frame[y1:y2, x1:x2]
        
        
        sharpeningKernel = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")

#filter2D is used to perform the convolution.
# The third parameter (depth) is set to -1 which means the bit-depth of the output image is the 
# same as the input image. So if the input image is of type CV_8UC3, the output image will also be of the same type
        output = cv2.filter2D(ROI, -1, sharpeningKernel)

        newROI = np.zeros(ROI.shape, ROI.dtype)
        alpha=3
        beta=60
        for y in range(ROI.shape[0]):
            for x in range(ROI.shape[1]):
                for c in range(ROI.shape[2]):
                    newROI[y,x,c] = np.clip(alpha*ROI[y,x,c] + beta, 0, 255)

        cv2.imshow('ROI',newROI)
#        print(x1,x2,y1,y2)
#    ROI = frame[y1:y2, x1:x2]
#    cv2.imshow('ROI',ROI)
#    if cv2.waitKey(10) & 0xFF == ord('a'):
#        break
        
   #     print('draw rectangle ',x,y,w,h)
    
    # detecting eyes
#    eyes = eyeCascade.detectMultiScale(imgGray)

    # drawing bounding box for eyes
#    for (ex, ey, ew, eh) in eyes:
#  

    cv2.imshow('face_detect', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyWindow('face_detect')