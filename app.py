from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = './model/model.h5'
saved_model = load_model(MODEL_PATH, compile=False)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        img = np.reshape(img, [1, 64, 64, 3])
        d = saved_model.predict(img)
        print(d)
        r = d[0][0]
        r = round(r) - 2
        if (r == 0):
            result = 'No DR'
        elif (r == 1):
            result = 'Mild DR'
        elif (r == 2):
            result = 'Moderate DR'
        elif (r == 3):
            result = 'Severe DR'
        else:
            result = 'Proliferative DR'

        return result
    else:
        return 'there is no scanned image attached'


if __name__ == '__main__':
    app.run()
