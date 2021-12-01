import cv2
import os
import mediapipe as mp


#def contour_bymp(img):
#    mpface = mp.solutions.face_detection
#
#    face_detect = mpface.FaceDetection()
#
#    mpdraw = mp.solutions.drawing_utils
#    results = face_detect.process(img)
#    if results.detections:
#        for id, detect in enumerate(results.detections):
#            box_pos = detect.location_data.relative_bounding_box
#
#            mpdraw.draw_detection(img, detect)
#
#    return
    

def contour_faces(img,frame = 0):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    #img = cv2.imread(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
#        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 6)
#        for(ex,ey,ew,eh) in eyes:
#            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Armin {}'.format(frame), (x+10, y+10), font, 1, (0, 255, 0), 2, cv2.LINE_AA, False)

    return faces
    
    
img = "image for faces.jpeg"

img = cv2.imread(img)

contour_faces(img)

cv2.imshow("image", img)
cv2.waitKey(0)



