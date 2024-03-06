import cv2
from simple_facerec import SimpleFacerec
import dlib
import csv
from datetime import datetime


existing_data = {}  
try:
    with open("attendance_data.csv", mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)  
        for row in reader:
            name, date_time, status = row
            existing_data[name] = {"date_time": date_time, "status": status}
except FileNotFoundError:
    print("Provided CSV file not found. Please make sure it exists.")
    exit()


cap = cv2.VideoCapture(0)  

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

processed_names = set()

def detect_blinks(landmarks, name):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    left_eye_aspect_ratio = blink_aspect_ratio(left_eye)
    right_eye_aspect_ratio = blink_aspect_ratio(right_eye)

    return left_eye_aspect_ratio, right_eye_aspect_ratio

def blink_aspect_ratio(eye):
    eye = list(map(lambda p: (p.x, p.y), eye))

    vertical_dist1 = ((eye[1][0] - eye[5][0]) ** 2 + (eye[1][1] - eye[5][1]) ** 2) ** 0.5
    vertical_dist2 = ((eye[2][0] - eye[4][0]) ** 2 + (eye[2][1] - eye[4][1]) ** 2) ** 0.5

    horizontal_dist = ((eye[0][0] - eye[3][0]) ** 2 + (eye[0][1] - eye[3][1]) ** 2) ** 0.5

    aspect_ratio = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    return aspect_ratio

def write_to_csv(name, present):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    existing_data[name]["date_time"] = current_time if present else ""
    existing_data[name]["status"] = "Present" if present else "Absent"
    
    with open("your_existing_data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, existing_data[name]["date_time"], existing_data[name]["status"]])

def update_status():
    for name in existing_data.keys():
        if name not in processed_names:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing_data[name]["date_time"] = current_time
            existing_data[name]["status"] = "Absent"
        else:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing_data[name]["date_time"] = current_time
            existing_data[name]["status"] = "Present"
    

    with open("your_existing_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)  
        for name, data in existing_data.items():
            writer.writerow([name, data["date_time"], data["status"]])




import atexit
atexit.register(update_status)

while True:
    ret, frame = cap.read()

    
    face_locations, face_name = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_name):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        
        landmarks = predictor(frame, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
        
        
        left_eye_aspect_ratio, right_eye_aspect_ratio = detect_blinks(landmarks.parts(), name)
        
        
        if left_eye_aspect_ratio < 0.2 and right_eye_aspect_ratio < 0.2 and name not in processed_names:
            print(f"{name} Attendance Marked!")
            write_to_csv(name, True)
            processed_names.add(name)
        else:
            write_to_csv(name, False)

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
