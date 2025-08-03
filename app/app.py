from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Dynamically build absolute path to the model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'combined_image_classifier.h5')
model = load_model(model_path)

# Class labels (adjust if needed)
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck', 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot', 'mnist_0', 'mnist_1',
    'mnist_2', 'mnist_3', 'mnist_4', 'mnist_5', 'mnist_6', 'mnist_7',
    'mnist_8', 'mnist_9', 'Cat', 'Dog'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_file = None  # define upfront to avoid UnboundLocalError

    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            img_path = os.path.join('static', image_file.filename)
            image_file.save(img_path)

            img = Image.open(img_path).resize((32, 32)).convert('RGB')
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            predicted_label = class_names[np.argmax(pred)]
            prediction = f"Predicted Class: {predicted_label}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
