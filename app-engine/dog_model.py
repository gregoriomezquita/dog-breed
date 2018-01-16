#
# This cell has all the dependencies
#
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image 
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.xception import decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
#from keras import backend as K

import cv2
import json

    
def load_dog_names(file_name):
    with open(file_name, 'r') as f: table= json.load(f)
    res= [str.replace(name, '_', ' ') for name in table] 
    return res

def path_to_tensor(img):
    #img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def get_dog_breed(img):
	feature= xception.predict(xception_preprocess_input(path_to_tensor(img)))
	prediction= model.predict(feature)
	max= np.argmax(prediction)
	accuracy= prediction[0, max]* 100
	return dog_names[max], accuracy

def face_detector(img):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    #img = cv2.imread(img)
    open_cv_image = np.array(img)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img):
	img50 = resnet50_preprocess_input(path_to_tensor(img))
    
	#global graph_resnet50
	with graph_resnet50.as_default():
		prediction = np.argmax(ResNet50_model.predict(img50))
	return ((prediction <= 268) & (prediction >= 151))

def predict(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	label= 'NONE'
	breed= ''
	acc= 0
	
	#global graph_xception
	with graph_xception.as_default():
		if face_detector(img):  
			label= 'HUMAN'
			breed, acc= get_dog_breed(img)
		elif dog_detector(img):
			label= 'DOG'
			breed, acc= get_dog_breed(img)
    	
	return {'label': label, 'breed': breed, 'accuracy': '%.3f' % acc}

def predict2(img_raw):
	img = image.load_img(img_raw, target_size=(299, 299))
	x= path_to_tensor(img)
	x = xception_preprocess_input(x)

	global graph_xception
	with graph_xception.as_default():
		preds = xception_full.predict(x)

	top3 = decode_predictions(preds,top=3)[0]

	predictions = [{'label': label, 'description': description, 'probability': probability * 100.0}
                    for label,description, probability in top3]
	return predictions

dog_names= load_dog_names('dog_names.json')

#xception_full = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
xception= Xception(weights='imagenet', include_top=False)
#xception= load_model('xception.h5')
graph_xception = tf.get_default_graph()

ResNet50_model = ResNet50(weights='imagenet')
graph_resnet50 = tf.get_default_graph()

with open("savedmodels/model_Xception.json", "r") as file: model= model_from_json(file.read())
model.load_weights('savedmodels/weights.best.Xception.hdf5') 
graph_model = tf.get_default_graph()

