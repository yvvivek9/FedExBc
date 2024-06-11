from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io
import os

import model

app = Flask(__name__)

image_height = 30
image_width = 20
image_folder = "../Model/ISIC-images"


# Define route to accept image input and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if image is uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    # Read the image file
    image_file = request.files['image']

    # Ensure the file is a valid image
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})

    # Read the image data
    image_bytes = image_file.read()

    # Convert image bytes to PIL Image object
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image
    image = image.resize((image_height, image_width))  # Resize the image to match your model's input shape
    image_array = np.asarray(image) / 255.0  # Normalize the image data

    # Make prediction
    prediction = model.predict_from_model(image_array)
    return jsonify({'prediction': prediction})


@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")


@app.route("/test/<image>", methods=['GET'])
def test(image):
    # Function to load and preprocess images
    def preprocess_image(image_path_local):
        with Image.open(image_path_local) as img:
            img = img.resize((image_width, image_height))
            img_array = np.asarray(img) / 255.0  # Normalize pixel values
        return img_array

    image_path = os.path.join(image_folder, image + ".jpg")  # Assuming images are in JPEG format
    if os.path.exists(image_path):
        try:
            processed_image = preprocess_image(image_path)
            prediction = model.predict_from_model(processed_image)
            return jsonify({'prediction': prediction})
        except:
            print("Failed to load: " + image_path)
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
