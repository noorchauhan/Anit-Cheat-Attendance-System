import cv2
from simple_facerec import SimpleFacerec
eye_cascade = cv2.CascadeClassifier('C:/Users/Shahnawaz/OneDrive/Desktop/Shanu College/face identity/haarcascade_eye_tree_eyeglasses.xml')
# Load Camera
cap = cv2.VideoCapture(0) #Open webcam 0 means 1 cam 


#Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


while True: 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,5,1,1)

    # Detect faces 
    face_locations, face_name = sfr.detect_known_faces(frame) 
    for face_loc, name in zip(face_locations, face_name):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3] # y1=top, x2= right, y2=bottom, x1=left
        # roi_face = gray[y:y+h,x:x+w]
        # roi_face_clr = img[y:y+h,x:x+w]
        roi_face = cv2.putText(frame,name, (x1, y1 -10), cv2.FONT_HERSHEY_DUPLEX, 1 , (0,0,200),2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

    eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
    if(len(eyes)>=2):
                if(first_read):
                    cv2.putText(img, "press s to begin", (100,100), cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                else:
                    print("Face Detected")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) #1 means it will go to next frame in 1 millisecond we have the video/camera in real time
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows

