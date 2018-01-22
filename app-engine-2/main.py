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
from PIL import Image, ImageOps


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		try:
			img = request.files.get('image')
			image= io.BytesIO(img.read())
			image = ImageOps.fit(image, [224, 224], Image.ANTIALIAS)
			res= '<html><body align= "center"><h1>Hello World!: POST</h1></body></html>'
		except Exception as e:
			return jsonify(status_code='400', msg='Bad Request: %s' % str(e)), 400
		return res
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
