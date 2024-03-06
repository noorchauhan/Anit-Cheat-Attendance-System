#importing libraries
import cv2
import face_recognition

img = cv2.imread("Messi1.webp") #CV2 iss reading image
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converting BGR image to RGB, mostly algorithm accept image in RGB format
img_encoding = face_recognition.face_encodings(rgb_img)[0] #Encoding image to extract facial features

img2 = cv2.imread("Messi1.webp")
rgb_img2 =cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2) #Comparing both image faces is same result will true else false
print("Result: ", result)


cv2.imshow("Img", img) #showing image
cv2.imshow("Img 2", img2)

cv2.waitKey(0) #If press 0 our image will freeze