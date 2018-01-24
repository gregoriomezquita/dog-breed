# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import logging

from flask import current_app, Flask, render_template, request, jsonify, send_from_directory
#from pillow import Image, ImageOps
from google.appengine.api import images
from google.appengine.ext import ndb
from google.appengine.api import app_identity
import os
import io
import base64
import json
#import request

token= app_identity.get_access_token('https://www.googleapis.com/auth/cloud-platform')[0]

app = Flask(__name__)

#url = 'https://ml.googleapis.com/v1/projects/{}/models/{}/versions/{}:predict&key={}'.format('udacity-190420','DogBreed','v1', auth)

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		try:
			img= request.files.get('image')
			image_data= img.read()
			image= images.Image(image_data=image_data)
			#image_bytes= io.BytesIO(image_stream)
			#img_b64= base64.b64encode(image_stream)
			#img_b64_str= str(img_b64)
			#image= images.Image(image_data=image_bytes)
			image.resize(width=224, height=224, crop_to_fit=True)
			#image.im_feeling_lucky()
			thumbnail= image.execute_transforms(output_encoding=images.JPEG)

			#img = request.files.get('image')
			#image= io.BytesIO(img.read())
			#image = ImageOps.fit(image, [224, 224], Image.ANTIALIAS)
			img_b64_str= str(base64.b64encode(image_data))
			#breed, acc= predict_raw(thumbnail)
			img_b64_str= str(base64.b64encode(thumbnail))
			breed, acc= None, 0.0
			prediction= {'label':'HUMAN', 'breed':token, 'accuracy':acc}
		except Exception as e:
			return jsonify(status_code='400', msg='Bad Request: %s' % str(e)), 400
		return render_template('view.html', image_data= img_b64_str, file_name= img.filename, content_type= img.content_type, prediction= prediction)
	else:
		return render_template('form.html')

@app.route('/favicon.ico')
def favicon():
	return send_from_directory(os.path.join(app.root_path, '.'), 'favicon.png', mimetype='image/png')

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500
# [END app]
