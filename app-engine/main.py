#
# Gregorio Mezquita
#

#import logging

from flask import current_app, Flask, render_template, request, jsonify, send_from_directory
#import json
import io
import os
import base64

import dog_model as model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		try:
			#data = request.get_json()['data']
			#data = base64.b64decode(data)
			#image = io.BytesIO(data)
			img = request.files.get('image')
			image_stream = img.read()
			image_bytes= io.BytesIO(image_stream)
			img_b64= base64.b64encode(image_stream)
			img_b64_str= str(img_b64, 'utf-8')
			#current_app.logger.info('Image: %s %s %s', type(img_b64_str), img_b64[:10], img_b64_str[:10])
			prediction = model.predict(image_bytes)
			if prediction['label']== 'DOG': prediction['label']= 'Dog breed: {}'.format(prediction['breed'])
			elif prediction['label']== 'HUMAN': prediction['label']= 'Human like {}'.format(prediction['breed'])
			else: prediction['label']= 'Neither human nor dog'
		except Exception as e:
			return jsonify(status_code='400', msg='Bad Request: %s' % str(e)), 400
		return render_template('view.html', image_data= img_b64_str, file_name= img.filename, content_type= img.content_type, prediction= prediction)
	return render_template('form.html')
	
@app.route('/favicon.ico')
def favicon():
	#return send_from_directory(os.path.join(app.root_path, 'www'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
	return send_from_directory(os.path.join(app.root_path, '.'), 'favicon.png', mimetype='image/png')


@app.errorhandler(500)
def server_error(e):
	logging.exception('An error occurred during a request.')
	return """
	An internal error occurred: <pre>{}</pre>
	See logs for full stacktrace.
	""".format(e), 500


if __name__ == '__main__':
	# This is used when running locally. Gunicorn is used to run the
	# application on Google App Engine. See entrypoint in app.yaml.
	app.run(host='127.0.0.1', port=8080, debug=True)
