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
from time import sleep
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
# Global variables
#
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

#
# Send a prediction request to cloud ml
#
def predict_raw(data):
    b64_x = base64.urlsafe_b64encode(data)
    input_instance = dict(inputs=b64_x)
    input_instance = json.loads(json.dumps(input_instance))
    
    request_body = {"instances": [input_instance]}
    request = ml.projects().predict(name=modelID, body=request_body)
    
    try:
        response = request.execute()
    except errors.HttpError as err:
        # Something went wrong with the HTTP transaction.
        # To use logging, you need to 'import logging'.
        print('There was an HTTP error during the request:')
        print(err._get_reason())

    if response.get('error') != None: print(response)
    else:
        predictions= response.get('predictions')[0].get('outputs')
        max= np.argmax(predictions)
        accuracy= predictions[max]* 100
        return dog_names[max], accuracy
    return '', 0.0
#
# Resize image and predict
#    
def predict_breed(image_data):
	image= images.Image(image_data= image_data)
	image.resize(width=224, height=224, crop_to_fit=True)
	thumbnail= image.execute_transforms(output_encoding=images.JPEG)

	global gbreed, gacc
	gbreed, gacc= predict_raw(thumbnail)
	#return predict_raw(thumbnail)
##################################
def task1(data):
	sleep(2)
	return
def process_image(image_data):
	#predict_image(image_data)
	breed_th= Thread(target= predict_breed, args= (image_data, ))
	breed_th.start()
	task1_th= Thread(target= task1, args= (image_data, ))
	task1_th.start()
	breed_th.join()
	task1_th.join()
	return gbreed, gacc
##################################
#
# POST Predict
#    
@app.route('/predict', methods=['POST'])
def predict():
	if 'image' not in request.files: return "No images", 500
	img= request.files.get('image')
	image_data= img.read()
	breed, acc= process_image(image_data)
	
	return app.response_class(
					response= json.dumps(dict(breed= breed, accuracy= acc)),
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
				breed, acc= process_image(r.content)
				img_b64_str= str(base64.b64encode(r.content))
				
				prediction= {'label':'DOG %s' % breed, 'breed':breed, 'accuracy':acc}
				return render_template('view.html', image_data= img_b64_str, file_name= img_url, content_type= r.headers.get('Content-Type'), prediction= prediction)
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
			breed, acc= process_image(image_data)
			img_b64_str= str(base64.b64encode(image_data))
			#img_b64_str= str(base64.b64encode(thumbnail))
			
			prediction= {'label':'DOG %s' % breed, 'breed':breed, 'accuracy':acc}
		except Exception as e:
			return jsonify(status_code='400', msg='Bad Request: %s' % str(e)), 400
		return render_template('view.html', image_data= img_b64_str, file_name= img.filename, content_type= img.content_type, prediction= prediction)
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
