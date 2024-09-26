import os
import cv2
import matplotlib.pyplot as plt

# Path to the results folder containing images
results_folder = r'C:\Users\Admin\Desktop\Personal Projects\Datasets\HIT-UAV dataset\hit-uav\results'

# List all image files in the results folder and sort them
image_files = sorted([f for f in os.listdir(results_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])[
              :20]  # Get the first 20 images

# Set up the plot for a grid of 4x5 (4 rows, 5 columns)
fig, axs = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('First 20 Frames from Results Folder', fontsize=16)

# Iterate through the first 20 images
for i, img_file in enumerate(image_files):
    # Read the image using OpenCV
    img = cv2.imread(os.path.join(results_folder, img_file))

    # Check if image is successfully loaded
    if img is None:
        print(f"Failed to load {img_file}")
        continue

    # Convert the image from BGR (OpenCV format) to RGB (matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Determine the row and column for the current image
    row = i // 5  # 4 rows (0 to 3)
    col = i % 5  # 5 columns (0 to 4)

    # Display the image in the grid
    axs[row, col].imshow(img_rgb)
    axs[row, col].axis('off')  # Hide axis
    axs[row, col].set_title(f"Frame {i + 1}")

# Adjust layout and display the grid
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
plt.show()
