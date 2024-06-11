import os
import pandas as pd

num_datasets = 8

# Check the number of benign and malignant data in each folder
for i in range(num_datasets):
    dataset_folder = f"Split/Part {i + 1}"
    metadata_path = os.path.join(dataset_folder, "metadata.csv")
    metadata = pd.read_csv(metadata_path)

    num_benign = metadata[metadata['benign_malignant'] == 'benign'].shape[0]
    num_malignant = metadata[metadata['benign_malignant'] == 'malignant'].shape[0]

    print(f"Dataset {i + 1}: Benign: {num_benign}, Malignant: {num_malignant}")
