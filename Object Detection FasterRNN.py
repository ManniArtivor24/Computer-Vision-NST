import os
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the image transform for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Define paths
image_folder = r"C:\Users\Admin\Desktop\Personal Projects\Datasets\HIT-UAV dataset\hit-uav\images\train"
results_folder = r"C:\Users\Admin\PycharmProjects\Computer-Vision-HDIDS\results"

# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Get list of images in the folder and limit to first 25 images
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))][:40]

# Define a threshold for confidence score
confidence_threshold = 0.3

# Loop through the first 25 images in the folder
for image_file in image_files:
    # Load and preprocess the image
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).convert("RGB")  # Convert to RGB as the model expects 3 channels

    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Run the model on the input image
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract the bounding boxes and labels
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # Load the image with OpenCV for display
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes on the image
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            # Extract box coordinates
            x1, y1, x2, y2 = box.int().numpy()

            # Draw rectangle with label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label.item()} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

    # Save the resulting image with detections to the results folder
    result_image_path = os.path.join(results_folder, f"detected_{image_file}")
    cv2.imwrite(result_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Save in BGR format as expected by OpenCV

print(f"Detection results for the first 25 frames have been saved to the '{results_folder}' folder.")
