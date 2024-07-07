import cv2
import os

import csv
import numpy as np
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

with open('data/names.pkl', 'rb') as f:
    Labels=pickle.load(f)

with open('data/face_data.pkl', 'rb') as f:
    Faces=pickle.load(f)

# Debugging print statements
print(f"Faces shape: {Faces.shape}")
print(f"Labels length: {len(Labels)}")

# Ensure that Faces and Labels have the same number of samples
if Faces.shape[0] != len(Labels):
    raise ValueError(f"Inconsistent number of samples: Faces ({Faces.shape[0]}) and Labels ({len(Labels)})")



knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(Faces,Labels)

#imgbg=cv2.imread("")

COL_NAMES=['NAME','TIME']




while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w,:]
        resize_img=cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)

        if resize_img.shape[1] != Faces.shape[1]:
            print(f"Error: Mismatched feature sizes. Expected {Faces.shape[1]}, but got {resize_img.shape[1]}")
            continue

        output = knn.predict(resize_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")

        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")
        
        # Create the file path
        file_path = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(file_path)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        #cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        attendance=[str(output[0]),str(timestamp)]

    cv2.imshow("frame",frame)

    k=cv2.waitKey(1)

    if k==ord('o'):
        time.sleep(5)

        if exist:
            with open("Attendance/Attendance_"+date+".csv", '+a') as f:
                writer=csv.writer(f)
                writer.writerow(attendance)
            f.close()

        else:
            with open("Attendance/Attendance_"+date+".csv", '+a') as f:
                writer=csv.writer(f)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            f.close()
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()