import cv2
import numpy as np
import os



##KNN CODE

def distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
	dist = []

	for i in range(train.shape[0]):

		ix = train[i, :-1]
		iy = train[i, -1]

		d = distance(test,ix)
		dist.append([d,iy])

	dk = sorted(dist, key=lambda x:x[0])
	dk = dk[:k]

	labels = np.array(dk)[:,-1]

	output = np.unique(labels,return_counts=True)
	index = np.argmax(output[1])
	return output[0][index]

###############
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
data_path = './data/'
face_data = []
labels = []

class_id = 0
names = {}


for fx in os.listdir(data_path):
	if fx.endswith('.npy'):
		#create the mapping of class_id with  name
		names[class_id] = fx[:-4]
		print("loaded "+fx)
		data_item = np.load(data_path + fx)
		face_data.append(data_item)

		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape);
print(face_labels.shape);

train_set = np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)

while True:
	ret,frame = cap.read()
	if ret==False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.5,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		out=knn(train_set,face_section.flatten())
		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
	
	cv2.imshow("FACES",frame)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()