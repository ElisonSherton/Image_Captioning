#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, redirect, url_for
import warnings
warnings.filterwarnings("ignore")
from utils import *


model_path = './artifacts/model_20_.h5'
tokenizer_path = './artifacts/tokenizer_vt2.pkl'

app = Flask(__name__)

# Path to save image to
app.config['IMAGE_UPLOADS_PATH'] = "./static/uploaded_img"

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

            # Generate prediction
            caption = utils.generate_caption(model_path = model_path,
                                             tokenizer_path = tokenizer_path,
                                             image_path = img_path)
            
            print(caption)
            
            return render_template("results.html", img_pth = img_path, preds = caption)

    return render_template("upload.html")



if __name__ == '__main__':
    app.run(debug = True)