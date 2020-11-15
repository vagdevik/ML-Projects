# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from app_helper import *

# Define a flask app
app = Flask(__name__)


@app.route("/")
def index():
  return render_template("index.html")


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	detected_objects = ""

	if request.method == 'POST':
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'static','uploads', secure_filename(f.filename))
		f.save(file_path)
		detected_objects = get_detected_image(file_path, f.filename)
		print(detected_objects)

	return render_template("uploaded.html", display_detection = f.filename, fname = f.filename, detected_objects=detected_objects) 


if __name__ == "__main__":
    app.run(debug=True)