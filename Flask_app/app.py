#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, redirect, url_for
import warnings
warnings.filterwarnings("ignore")
from utils import *


model_path = '/content/drive/My Drive/model_20_.h5'
tokenizer_path = './artifacts/tokenizer_vt2.pkl'
img_abs_path = '/content/drive/My Drive/2015-Must-have-Riding-Gear.jpg'

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        img_path = None
        image = request.files['image']

        if image:
            # Create image path using os
            img_path = os.path.join(app.config["IMAGE_UPLOADS_PATH"], image.filename)

            # Save the image
            image.save(img_path)

            # If the image wasn't already uploaded, add the image path to our app.config
            if not img_path in app.config['ALL_IMAGES']:
                app.config['ALL_IMAGES'][image.filename] = img_path

            # If we've already made the prediction for this image, retrieve them, else actually predict
            predictions = None
            if not image.filename in app.config['IMAGE_PREDICTIONS']:
                predictions = get_pred(img_path)
                app.config['IMAGE_PREDICTIONS'][image.filename] = predictions
            else:
                predictions = app.config['IMAGE_PREDICTIONS'][image.filename]
            
            return render_template("results.html", img_pth = img_path, preds = predictions)

    return render_template("upload.html")



if __name__ == '__main__':
    app.run(debug = True)