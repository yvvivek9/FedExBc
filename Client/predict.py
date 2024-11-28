import torch
import os
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Path to weights
WEIGHTS_PATH = "fine_tuned_model.pth"
LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def model_prediction(img_path):
    # Load the image
    # img_path = os.path.join("..", "Dataset", "ISIC_2019_Training_Input", "ISIC_2019_Training_Input", "ISIC_0000001.jpg")
    image = Image.open(img_path).convert("RGB")

    # Apply the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load the model
    model = models.resnet50()  # Pretrained set to False for fine-tuned weights
    model.fc = torch.nn.Linear(model.fc.in_features, 9)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()

    # Pass the image through the model
    with torch.no_grad():
        outputs = model(image_tensor)  # Raw logits

    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(outputs)

    # Predicted classes
    predicted_classes = (probabilities > 0.5).int()
    predicted_label = predicted_classes.argmax(dim=1).item()

    print("Probabilities:", probabilities)
    print("Predicted Class:", LABELS[int(predicted_label)])

    # Grad-CAM implementation
    # Specify the target layer for Grad-CAM
    target_layer = model.layer4[-1]

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Preprocess image for Grad-CAM
    input_tensor = preprocess_image(np.array(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Generate Grad-CAM
    targets = [ClassifierOutputTarget(predicted_label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # Generate CAM
    grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension

    # Overlay Grad-CAM on the original image
    rgb_image = np.array(image) / 255.0  # Scale to [0, 1] for visualization
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    # Visualize the Grad-CAM
    plt.figure(figsize=(10, 10))
    plt.imshow(cam_image)
    plt.axis("off")
    plt.title(f"Grad-CAM for Class {LABELS[int(predicted_label)]}")
    plt.show()

