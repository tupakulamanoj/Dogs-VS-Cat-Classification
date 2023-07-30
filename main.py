from flask import Flask, render_template, request, flash
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import  img_to_array
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import os

app = Flask(__name__)


pred_model = load_model('cats_vs_dog.h5')


def preprocess_image(img):
    img = Image.open(img)
    img.save("./static/image/img.jpeg")
    img = img.resize((256, 256))  # Resize to match the input size of the model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':

        img = request.files['file']
        img_array = preprocess_image(img)
        pred = np.argmax(pred_model.predict(img_array))
        if pred == 0:
            return render_template('index2.html',cat='cat')
        else:
            return render_template('index2.html',cat='dog')


    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
