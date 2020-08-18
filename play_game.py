import cv2
from tensorflow import keras
import random
import numpy as np

class Game():
	def play(self, player_1, comp_player):
		player_1 = int(player_1)
		comp_player = int(comp_player)
		if player_1 == comp_player:
			return "TIE"
		else:
			if comp_player - player_1 == 1 or comp_player - player_1 == -2:
				return "The Computer won"
			else: 
				return "You won"

def show_icon(frame, computer_image_path):
	if computer_image_path != '':
		icon = cv2.imread(computer_image_path)
		icon = cv2.resize(icon, (400, 400))
		frame[100:500, 800:1200] = icon		



model_path = 'saved_model/rps_model.h5'
choices_p = 'choices/'
## Open Web Cam
cap = cv2.VideoCapture(0)

RPS = {'0': 'rock', '1': 'paper','2': 'scissors', '3': 'none'}

model = keras.models.load_model(model_path)
previous_move = '3'
computer_move = '3'
computer_image_path = ''
winner = 'Press Enter to begin playing'
game = Game()
while True:
	ret, frame = cap.read()
	# rectangle for user to play (400 x 400) 
	cv2.rectangle(frame, (100, 100), (500, 500), (255, 0, 0), 2)
	# for computer to play (400 x 400) 
	cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)
	
	# extracting
	roi = frame[100:500, 100:500]
	# normalizing
	img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) 
	# resizing image
	img = cv2.resize(img, (300, 300))
	
	# predict
	prediction = model.predict(np.array([img]))
	user_move = str(np.argmax(prediction[0]))
	if  user_move != '3'and k == 13:
		computer_move = random.randint(0, 2)
		winner = game.play(user_move, computer_move)
		computer_image_path = choices_p + str(computer_move) + '.png'
		show_icon(frame, computer_image_path)

		
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, "User: " + RPS[user_move], (100, 96), font, 1, (0, 0,0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Winner: " + winner,(50, 48), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)   
	cv2.putText(frame, "Computer :" + RPS[str(computer_move)], (800, 96), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
	show_icon(frame, computer_image_path)
	cv2.imshow("Rock Paper Scissors", frame)
	
	
	
	k = cv2.waitKey(10)
	if k == ord('q'):
		break  
cap.release()
cv2.destroyAllWindows()
    
    



