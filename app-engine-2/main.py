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
import time
import numpy as np
from google.appengine.api import images
from google.appengine.api import app_identity
from google.appengine.api import urlfetch
#
# from a python 2 environment:
# pip install --upgrade -t lib requirements.txt
#
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors


app = Flask(__name__)

with open('dog_names.json', 'r') as f: dog_names= json.load(f)

left= 0
auth= []

def get_token():
	global left, auth
	
	if left <= 0:
		auth= app_identity.get_access_token(['https://www.googleapis.com/auth/cloud-platform'])
		left= auth[1]- time.time()
	return auth[0]
	
PROJECTID = 'mezquita-dog-breed'
projectID = 'projects/{}'.format(PROJECTID)
modelName = 'DogBreed'
modelID = '{}/models/{}'.format(projectID, modelName)

credentials = GoogleCredentials.get_application_default()
ml = discovery.build('ml', 'v1', credentials=credentials)
	
print('----------------', type(credentials))
	

#token= app_identity.get_access_token(['https://www.googleapis.com/auth/cloud-platform',
																			#'https://www.googleapis.com/auth/cloud-platform.read-only',
																			#'https://www.googleapis.com/auth/cloud-platform',
																			#'https://www.googleapis.com/auth/prediction']
																			#)[0]
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
        #print(type(predictions), predictions)
        max= np.argmax(predictions)
        accuracy= predictions[max]* 100
        return dog_names[max], accuracy
    return '', 0.0
    
def predict_rest(data):
	try:
		b64 = base64.urlsafe_b64encode(data)
		input_instance = dict(inputs=b64)
		input_instance = json.loads(json.dumps(input_instance))
		request_body = {"instances": [input_instance]}
		
 		result = urlfetch.fetch(
        url = 'https://ml.googleapis.com/v1/projects/{}/models/{}/versions/{}:predict'.format('udacity-190420','DogBreed','v1'),
        payload=json.dumps(request_body),
        method=urlfetch.POST,
        headers={
            'Content-Type': 'application/json',
            'Authorization': "Bearer %s" % get_token()
              }
        )
		response= json.loads(result.content)
		if response.get('error') != None: print(response)
		else:
			predictions= response.get('predictions')[0].get('outputs')
			max= np.argmax(predictions)
			accuracy= predictions[max]* 100
			return dog_names[max], accuracy
		return '', 0.0
	except urlfetch.Error: logging.exception('Caught exception fetching url')
	
	return '', 0.0

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		try:
			img= request.files.get('image')
			image_data= img.read()
			
			image= images.Image(image_data= image_data)
			image.resize(width=224, height=224, crop_to_fit=True)
			image.im_feeling_lucky()
			thumbnail= image.execute_transforms(output_encoding=images.JPEG)

			img_b64_str= str(base64.b64encode(image_data))
			#img_b64_str= str(base64.b64encode(thumbnail))
			breed, acc= predict_raw(thumbnail)
			#breed, acc= predict_rest(thumbnail)
			prediction= {'label':'DOG %s' % breed, 'breed':breed, 'accuracy':acc}
		except Exception as e:
			return jsonify(status_code='400', msg='Bad Request: %s' % str(e)), 400
		return render_template('view.html', image_data= img_b64_str, file_name= img.filename, content_type= img.content_type, prediction= prediction)
	else:
		return render_template('form.html')

@app.route('/_ah/auth', methods=['GET', 'POST'])
def auth():
	#return app_identity.get_access_token('https://www.googleapis.com/auth/cloud-platform')[0]
	return get_token()

@app.route('/favicon.ico')
def favicon():
	return send_from_directory(os.path.join(app.root_path, '.'), 'favicon.png', mimetype='image/png')

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
# [END app]
