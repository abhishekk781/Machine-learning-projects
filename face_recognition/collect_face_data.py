import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
data_path = './data/'
face_data = []
skip = 0
file_name = raw_input("Enter the name: ")

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces =face_cascade.detectMultiScale(frame,1.3,2)
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	face_section = frame
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip+=1
		if (skip%10==0):
			face_data.append(face_section)
	cv2.imshow("face",frame)
	cv2.imshow("face section", face_section)
	
	print(len(face_data))

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(data_path+file_name+'.npy',face_data)



cap.release()
cv2.destroyAllWindows()