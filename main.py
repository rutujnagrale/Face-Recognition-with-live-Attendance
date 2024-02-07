import cv2
import numpy as np
import face_recognition


imgHrucha = face_recognition.load_image_file('ImagesBasic/rucha.jpeg')
imgHrucha = cv2.cvtColor(imgHrucha,cv2.COLOR_BGR2RGB)
imgNadaf = face_recognition.load_image_file('ImagesBasic/sumaiya.jpeg')
imgNadaf = cv2.cvtColor(imgNadaf,cv2.COLOR_BGR2RGB)
imgNagrale = face_recognition.load_image_file('ImagesBasic/aditya.jpeg')
imgNagrale = cv2.cvtColor(imgNagrale,cv2.COLOR_BGR2RGB)
imgrutuj = face_recognition.load_image_file('ImagesBasic/Rutuj .jpeg')
imgrutuj = cv2.cvtColor(imgrutuj,cv2.COLOR_BGR2RGB)

faceLocHrucha = face_recognition.face_locations(imgHrucha)[0]
encodeHrucha = face_recognition.face_encodings(imgHrucha)[0]
cv2.rectangle(imgHrucha,(faceLocHrucha[3],faceLocHrucha[0]),(faceLocHrucha[1],faceLocHrucha[2]),(255,0,255),2)

faceLocNadaf = face_recognition.face_locations(imgNadaf)[0]
encodeNadaf = face_recognition.face_encodings(imgNadaf)[0]
cv2.rectangle(imgNadaf,(faceLocNadaf[3],faceLocNadaf[0]),(faceLocNadaf[1],faceLocNadaf[2]),(255,0,255),2)

faceLocNagrale = face_recognition.face_locations(imgNagrale)[0]
encodeNagrale = face_recognition.face_encodings(imgNagrale)[0]
cv2.rectangle(imgNagrale,(faceLocNagrale[3],faceLocNagrale[0]),(faceLocNagrale[1],faceLocNagrale[2]),(255,0,255),2)

faceLocrutuj = face_recognition.face_locations(imgrutuj)[0]
encoderutuj = face_recognition.face_encodings(imgrutuj)[0]
cv2.rectangle(imgrutuj,(faceLocrutuj[3],faceLocrutuj[0]),(faceLocrutuj[1],faceLocrutuj[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodeHrucha],encodeNadaf)
faceDis = face_recognition.face_distance([encodeHrucha],encodeNadaf)
print(results,faceDis)
cv2.putText(imgNadaf,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Hrucha Nagre',imgHrucha)
cv2.imshow('Sumaiyya Nadaf',imgNadaf)
cv2.imshow('Aditya Nagrale',imgNagrale)
cv2.imshow('Rutuj Nagrale',imgrutuj)
cv2.waitKey(0)