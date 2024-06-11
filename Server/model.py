from tensorflow.keras.models import model_from_json
import numpy as np
import json


labels_path = "../Model/SaveFile/encoded_labels.json"
json_path = "../Model/SaveFile/model.json"
h5_path = "../Model/SaveFile/model.weights.h5"


def load_model():
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(h5_path)

    # evaluate loaded model on test data
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print("Loaded model from disk")
    return loaded_model


model = load_model()


def predict_from_model(image):
    predictions = model.predict(np.array([image]))
    predicted_indices = np.argmax(predictions, axis=1)
    with open(labels_path, 'r') as f:
        labels = json.load(f)
        print(labels[predicted_indices[0]])
        return labels[predicted_indices[0]]
