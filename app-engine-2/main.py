#
# Gregorio Mezquita 
# Jan 2018
#
# To run the app in a local envirenment(python 2):
# dev_appserver.py app.yaml
#
# To deploy in Google:
# gcloud app deploy [--project PROJECT-ID] [-v VERSION]
#
# [START app]
import logging
from flask import current_app, Flask, render_template, request, jsonify, send_from_directory
import os
import io
import base64
import json
from threading import Thread
import urllib
import numpy as np
from google.appengine.api import images
from google.appengine.api import urlfetch
#
# from a python 2 environment:
# pip install --upgrade -t lib requirements.txt
#
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors


app = Flask(__name__)
#
# Global variables to use with threads
#
gIsHuman= False
gIsDog= False
gbreed= ''
gacc= 0.0

# Create dog names table from file
with open('dog_names.json', 'r') as f: dog_names= json.load(f)

	
projectId= 'mezquita-dog-breed'
modelName = 'DogBreed'
modelVersion= 'v1'

modelID = 'projects/{}/models/{}'.format(projectId, modelName)
if modelVersion is not None: modelID += '/versions/{}'.format(modelVersion)

ml= discovery.build('ml', 'v1', credentials= GoogleCredentials.get_application_default() )
vision = discovery.build('vision', 'v1', credentials= GoogleCredentials.get_application_default() )
#
# Search for breed examples
#
def breed_examples(breed):
	query = 'dog '+ breed
	params={
        'count': 5,
        'q': query,
        't': 'images',
        'safesearch': 1,
        'locale': 'en_US'
    }
	url= "https://api.qwant.com/api/search/images?"+ urllib.urlencode(params)
	r = urlfetch.fetch(url,
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
    }
	)

	response = json.loads(r.content).get('data').get('result').get('items')
	urls = [r.get('media') for r in response]
	
	return urls
#
# Returns a squared image of (224, 224) pixels
#	
def resize_image(image_data):
	image= images.Image(image_data= image_data)
	image.resize(width=224, height=224, crop_to_fit=True)
	return image.execute_transforms(output_encoding=images.JPEG)
	
#
# Predict dog breed.
# Send a prediction request to cloud ml
#    
def predict_breed(image_data):
	global gbreed, gacc
	gbreed= ''
	gacc= 0.0
	
	thumbnail= resize_image(image_data)

	b64_x= base64.urlsafe_b64encode(thumbnail)
	input_instance=  dict(inputs= b64_x)
	request = ml.projects().predict( name= modelID, body= dict(instances= [input_instance]) )
	response = request.execute()
	if response.get('error') == None:
		predictions= response.get('predictions')[0].get('outputs')
		max= np.argmax(predictions)
		gbreed= dog_names[max]
		gacc= predictions[max]* 100
	
	return gbreed, gacc
#
# Predict human
#
def predict_human(image_data, threshold= 0.7):
	thumbnail= resize_image(image_data)
	service_request = vision.images().annotate(
     body={ 'requests': [{
         'image': { 'content': base64.urlsafe_b64encode(thumbnail) },
         'features': [{ 'type': 'FACE_DETECTION' }]
        	}] })
	response = service_request.execute()
    
	global gIsHuman
	gIsHuman= False
	accs= [0]
	for results in response['responses']:
		if 'faceAnnotations' in results: 
			faces= results['faceAnnotations']
			for face in faces:
				if 'detectionConfidence' in face: 
					accs.append(face.get('detectionConfidence'))
	acc= max(accs)
	gIsHuman= acc > threshold
	return gIsHuman, acc
	
#
# Predict dog
#
def predict_dog(image_data):
	thumbnail= resize_image(image_data)
	service_request = vision.images().annotate(
     body={'requests': [{
         'image': { 'content': base64.urlsafe_b64encode(thumbnail) },
         'features': [{ 'type': 'LABEL_DETECTION', }]
        }] })
	global gIsDog
	gIsDog= False
	acc= 0.0
	response = service_request.execute()
	for results in response['responses']:
		if 'labelAnnotations' in results:
			for annotations in results['labelAnnotations']:
				desc= annotations['description']
				if 'dog' in desc: 
					gIsDog= True
					break
				elif 'dog breed' in desc: 
					gIsDog= True
					break
				elif 'dog breed group' in desc: 
					gIsDog= True
					break
			acc= annotations['score']

	return gIsDog	, acc		
               
##################################
# Main image process
# 3 predictions in parallel
##################################
def process_image(image_data):
	#predict_image(image_data)
	breed_th= Thread(target= predict_breed, args= (image_data, ))
	breed_th.start()
	label_th= Thread(target= predict_dog, args= (image_data, ))
	label_th.start()
	human_th= Thread(target= predict_human, args= (image_data, ))
	human_th.start()
	breed_th.join()
	label_th.join()
	human_th.join()
	global gIsHuman, gIsDog, gbreed, gacc
	acc= '{:.2f} %'.format(gacc)
	
	if gIsHuman:
		label= 'Human'
	elif gIsDog:
		label= 'Dog'
	else:
		label= 'Not human nor dog'
		gbreed= ''
		acc= ''
	return label, gbreed, acc
##################################
##################################
#
# POST Predict is dog
#    
@app.route('/dog', methods=['POST'])
def dog():
	if 'image' not in request.files: return "No images", 500
	img= request.files.get('image')
	image_data= img.read()
	h, acc= predict_dog(image_data)
	
	return app.response_class(
					response= json.dumps(dict(result= h, accuracy= acc * 100)),
					status=200,
					mimetype='application/json'
				)
#
# POST Predict is human
#    
@app.route('/human', methods=['POST'])
def human():
	if 'image' not in request.files: return "No images", 500
	img= request.files.get('image')
	image_data= img.read()
	h, acc= predict_human(image_data)
	
	return app.response_class(
					response= json.dumps(dict(result= h, accuracy= acc * 100)),
					status=200,
					mimetype='application/json'
				)
#
# POST Predict breed
#    
@app.route('/breed', methods=['POST'])
def breed():
	if 'image' not in request.files: return "No images", 500
	img= request.files.get('image')
	image_data= img.read()
	breed, acc= predict_breed(image_data)
	
	return app.response_class(
					response= json.dumps(dict(breed= breed, accuracy= acc)),
					status=200,
					mimetype='application/json'
				)
#
# POST Predict dog breed
#    
@app.route('/predict', methods=['POST'])
def predict():
	if 'image' not in request.files: return "No images", 500
	img= request.files.get('image')
	image_data= img.read()
	label, breed, acc= process_image(image_data)
	
	return app.response_class(
					response= json.dumps(dict(result= label, breed= breed, accuracy= acc)),
					status=200,
					mimetype='application/json'
				)
#
# Random images
#	
@app.route('/random', methods=['GET'])
def random():
	try:
		r= urlfetch.fetch('https://dog.ceo/api/breeds/image/random')
		if r.status_code == 200:
			img_url= json.loads(r.content).get('message')
			r = urlfetch.fetch(img_url)
			if r.status_code == 200:
				label, breed, acc= process_image(r.content)
				img_b64_str= str(base64.b64encode(r.content))
				if breed != '': examples= breed_examples(breed)
				else: examples= []
				presentation= dict(image_data= img_b64_str, file_name= img_url, content_type= r.headers.get('Content-Type'), examples=examples)
				if label== 'Human': label= "Human like '%s'" % breed
				if label== 'Dog': label= "Dog: '%s'" % breed
				prediction= dict(label= label, breed= breed, accuracy= acc)
				return render_template('carousel.html', presentation= presentation, prediction= prediction)
		return r.text
	except urlfetch.Error as e:
		return str(e), 500
	return "Unknown error", 500
#
# Home
#
@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		if 'image' not in request.files: return "No images", 500
		try:
			img= request.files.get('image')
			image_data= img.read()
			label, breed, acc= process_image(image_data)
			img_b64_str= str(base64.b64encode(image_data))
			if breed != '': examples= breed_examples(breed)
			else: examples= []
			presentation= dict(image_data= img_b64_str, file_name= img.filename, content_type= img.content_type, examples=examples)
			if label== 'Human': label= "Human like '%s'" % breed
			if label== 'Dog': label= "Dog: '%s'" % breed
			prediction= dict(label= label, breed= breed, accuracy= acc)
		except Exception as e:
			return jsonify(status_code='400', msg='Bad Request: %s' % str(e)), 400
		return render_template('carousel.html', presentation= presentation, prediction= prediction)
	else:
		return render_template('form.html', action= 'Select')

@app.route('/favicon.ico')
def favicon():
	return send_from_directory(os.path.join(app.root_path, '.'), 'favicon.png', mimetype='image/png')

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
# [END app]
