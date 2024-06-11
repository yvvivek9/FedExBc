import pandas as pd
import numpy as np
import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

IMAGE_FOLDER = "Dataset/Split/Part 1"
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


def process_data(image_height, image_width):
    # Load the dataset
    data = pd.read_csv(IMAGE_FOLDER + "/metadata.csv")

    # Filter data to get 500 benign and 500 malignant entries
    benign_data = data[data['benign_malignant'] == 'benign'].sample(n=1100)
    malignant_data = data[data['benign_malignant'] == 'malignant']

    # Combine benign and malignant data
    data_sample = pd.concat([benign_data, malignant_data])

    # Select relevant columns (isic_id and benign_malignant)
    data_sample = data_sample[['isic_id', 'benign_malignant']]

    # Function to load and preprocess images
    def preprocess_image(image_path_local):
        with Image.open(image_path_local) as img:
            img = img.resize((image_width, image_height))
            img_array = np.asarray(img) / 255.0  # Normalize pixel values
        return img_array

    # Iterate through rows to load and preprocess images
    images = []
    labels = []
    j = 0
    print()
    for index, row in data_sample.iterrows():
        image_path = os.path.join(IMAGE_FOLDER, row['isic_id'] + ".jpg")  # Assuming images are in JPEG format
        print(LINE_UP, end=LINE_CLEAR)
        if os.path.exists(image_path):
            try:
                images.append(preprocess_image(image_path))
                labels.append(row['benign_malignant'])
                j = j + 1
                print(f"Pre-processing image count: {j}")
            except:
                print("Failed to add: " + image_path, end="\n\n")
                pass
        else:
            print(f"Image {row['isic_id']} not found", end="\n\n")
    print("Image pre-processing done")

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Save label encoder to a JSON file
    with open('SaveFile/encoded_labels.json', 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)

    print("Data processing done")

    return X_train, X_test, y_train_encoded, y_test_encoded
