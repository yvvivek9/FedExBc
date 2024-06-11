import os
import time
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# Read metadata
metadata_path = "ISIC-images/metadata.csv"
metadata = pd.read_csv(metadata_path)

# Split metadata into parts
m_part_1, m_part_2 = train_test_split(metadata, test_size=0.5, stratify=metadata["benign_malignant"])
m_part_11, m_part_12 = train_test_split(m_part_1, test_size=0.5, stratify=m_part_1["benign_malignant"])
m_part_21, m_part_22 = train_test_split(m_part_2, test_size=0.5, stratify=m_part_2["benign_malignant"])
m_part_111, m_part_112 = train_test_split(m_part_11, test_size=0.5, stratify=m_part_11["benign_malignant"])
m_part_121, m_part_122 = train_test_split(m_part_12, test_size=0.5, stratify=m_part_12["benign_malignant"])
m_part_211, m_part_212 = train_test_split(m_part_21, test_size=0.5, stratify=m_part_21["benign_malignant"])
m_part_221, m_part_222 = train_test_split(m_part_22, test_size=0.5, stratify=m_part_22["benign_malignant"])

metadata_splits = [m_part_111, m_part_112, m_part_121, m_part_122, m_part_211, m_part_212, m_part_221, m_part_222]

# Define number of datasets and create folders
num_datasets = len(metadata_splits)
for i in range(num_datasets):
    os.makedirs(f"Split/Part {i + 1}", exist_ok=True)

# Copy images to respective folders and create metadata.csv
for i in range(num_datasets):
    print(f"\nDataset {i + 1} starting")
    dataset_folder = f"Split/Part {i + 1}"
    dataset_metadata = metadata_splits[i]
    dataset_metadata.to_csv(os.path.join(dataset_folder, "metadata.csv"), index=False)

    j = 0
    print()  # Extra line to compensate for LINE_CLEAR
    for _, row in dataset_metadata.iterrows():
        image_name = row['isic_id'] + ".jpg"  # Assuming images have .jpg extension
        source_path = os.path.join("ISIC-images", image_name)
        target_path = os.path.join(dataset_folder, image_name)
        shutil.copyfile(source_path, target_path)
        j = j + 1
        print(LINE_UP, end=LINE_CLEAR)
        print(f"Image Count: {j}")
    print(f"Dataset {i + 1} done")
