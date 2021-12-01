from djitellopy import tello
import cv2
import mediapipe as mp

#Camera capture dimensions
width = 320
height = 240

#0 = drone should fly, 1 = drone won't
startCounter = 0

#Connect to Tello
#me = tello.Tello()
#me.connect()
#me.for_back_velocity = 0
#me.left_right_velocity = 0
#me.up_down_velocity = 0
#me.yaw_velocity = 0
#me.speed = 0

#Display whether battery is good for flight
#print(me.get_battery()

#me.streamoff()
#me.streamon()

def contour_bymp(img):

    count_faces = 0

    mpface = mp.solutions.face_detection

    face_detect = mpface.FaceDetection()

    mpdraw = mp.solutions.drawing_utils
    results = face_detect.process(img)
    if results.detections:
    
        count_faces += 1
        
        for id, detect in enumerate(results.detections):
            box_pos = detect.location_data.relative_bounding_box

            mpdraw.draw_detection(img, detect)

    return count_faces
    
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

capture = cv2.VideoCapture(0)

count_faces = 0

while True:

    istrue, img = capture.read()
#    frame_read = me.get_frame_read()
#    myFrame = frame_read.frame
#    img = cv2.resize(myFrame, (width, height))

    #Initial takeoff, if drone hasn't already (startCounter)
#    if startCounter == 0:
#        me.takeoff()
#        #20 centimeters to the left
#        me.move_left(20)
#        #90 degrees clockwise
#        me.rotate_clockwise(90)
#        startCounter = 1

    count_faces = contour_faces(img)

    cv2.imshow("MyResult", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        me.land()
        break
        
print(count_faces)
        
