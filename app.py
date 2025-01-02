from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import os
from PIL import Image
from datetime import datetime
# from flask_cors import CORS

app = Flask(__name__)

# Load MobileNetV2 model
model_mobilenetv2 = load_model("MobileNetV2-pest-72.50.h5")

UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            resp = jsonify({'message': 'No image in the request'})
            resp.status_code = 400
            return resp

        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = datetime.now().strftime("%d%m%y-%H%M%S") + ".png"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            resp = jsonify({'message': 'File type is not allowed'})
            resp.status_code = 400
            return resp

        # Prepare image for prediction
        img = load_img(file_path, target_size=(224, 224))
        x = img_to_array(img)
        x = x / 127.5 - 1  # Normalize the image
        x = np.expand_dims(x, axis=0)

        # Predict
        prediction_array = model_mobilenetv2.predict(x)

        # Prepare API response
        class_names = ['daun_bercak', 'daun_karat', 'daun_sehat', 'embun_jelaga']

        return render_template("index.html", 
                               img_path=file_path,
                               prediction=class_names[np.argmax(prediction_array)],
                               confidence='{:2.0f}%'.format(100 * np.max(prediction_array)))

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)