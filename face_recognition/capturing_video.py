import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
	ret,frame = cap.read()
	frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(frame2, 1.5, 2)
	print(len(faces))

	for (x,y,w,h) in faces:
		print(x,y)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow("Abhishek",frame)
	#cv2.imshow("gray abhi",frame2)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()