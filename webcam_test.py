from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from keras.models import model_from_json
import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.preprocessing import image


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-j", "--json", required=True,
	help="path to json model file")
ap.add_argument("-c", "--config", required=True,
	help="path to h5 model file")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  json_file = open(args['json'], 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(args['config'])

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Loaded model from disk")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#class labels
train_names = ['M', 'del', 'A', 'C', 'K', 'I', 'H', 'P', 'L', 'N', 'U', 'Q', 'Y', 'Z', 'nothing', 'O', 'J', 'X', 'R', 'F', 'W', 'T', 'E', 'B', 'space', 'G', 'S', 'V', 'D']

#size of rectangle
(x1,y1,x2,y2) = (50,50,300,300)

# loop over the frames from the video stream
while True:
	#grab the frame from the threaded video stream and resize it
	#to have a maximum width of 600 pixels
	frame = vs.read()
	frame = cv2.flip(frame, 1)
	###
	#PREPROCESS IMAGES
	###
	frame = imutils.resize(frame, width=600)
    #draw square on screen
	frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    #show only what is in rectangle
	roi = frame[y1:y2, x1:x2]
	cv2.imshow('Rect', roi)
	#process what is inside rectangle
	roi = cv2.resize(roi, (200,200))
	x = image.img_to_array(roi)
	x = np.expand_dims(x, axis=0)
	#predict on what is inside rectangle
	y_prob = loaded_model.predict(x)
	y_classes = y_prob.argmax(axis=-1)
	prediction =  sorted(train_names)[int(y_classes)]
	text = 'Predicted: ' + prediction
	print(text)
	#add label above box
	cv2.putText(frame, text, (x1, int(y1*0.65)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 3)
	#show frame
	cv2.imshow("Frame", frame)
    #close out
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

#close windows and stop video stream
cv2.destroyAllWindows()
vs.stop()