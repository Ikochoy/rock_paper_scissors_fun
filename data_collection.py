import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
move_id = input('Enter move_id: ')
sample_no = 0
while True:
	ret, frame = cap.read()
	cv2.rectangle(frame, (100, 100), (500, 500), (255, 0, 0), 2)
	roi = frame[100:500, 100:500]
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("training_images/"+ str(move_id) + "/" + str(sample_no) + ".jpg", gray)
	sample_no += 1
	cv2.imshow("Collecting images", frame)
	cv2.waitKey(100)
	if sample_no == 200:
		break
cap.release()
cv2.destroyAllWindows()
