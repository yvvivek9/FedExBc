from tensorflow.keras.models import model_from_json
from PIL import Image
import os
import numpy as np
import json

import data_preprocess
import model_prepare

image_height = 300
image_width = 200

# Step 1: Pre-process data
X_train, X_test, y_train, y_test = data_preprocess.process_data(image_height, image_width)

# Step 2: Model Architecture
model = model_prepare.prepare_model(image_height, image_width)

# Step 3: Training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
print("Model trained")

# Step 4: Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Step 5: Save the model
# serialize model to JSON
model_json = model.to_json()
with open("SaveFile/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("SaveFile/model.weights.h5")
print("Saved model to disk")

# Step 6: Check saved model accuracy
json_file = open("SaveFile/model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("SaveFile/model.weights.h5")
# evaluate loaded model on test data
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
test_loss1, test_accuracy1 = loaded_model.evaluate(X_test, y_test)
print(f'Saved model Accuracy: {test_accuracy1}')


# Step 7: Loop run prediction
# Function to load and preprocess images
def preprocess_image(image_path_local):
    with Image.open(image_path_local) as img:
        img = img.resize((image_width, image_height))
        img_array = np.asarray(img) / 255.0  # Normalize pixel values
        return img_array


while 1:
    image = input("Enter image ID: ")
    image_path = os.path.join("Dataset", "ISIC-images", image + ".jpg")  # Assuming images are in JPEG format
    if os.path.exists(image_path):
        try:
            processed_image = preprocess_image(image_path)
            predictions = model.predict(np.array([processed_image]))
            predicted_indices = np.argmax(predictions, axis=1)
            with open("SaveFile/encoded_labels.json", 'r') as f:
                labels = json.load(f)
                print(labels[predicted_indices[0]])
        except:
            print("Failed to load: " + image_path)
            pass
