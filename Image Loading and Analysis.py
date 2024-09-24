import os
import cv2
import matplotlib.pyplot as plt

# Paths to the image directory
image_path = r"C:\Users\Admin\Desktop\Personal Projects\Datasets\HIT-UAV dataset\hit-uav\images\train"

# List all image files in the directory
image_files = sorted(os.listdir(image_path))[:15]  # Get only the first 15 images

# Set up the plot for a grid of 5x6 (5 rows, 6 columns)
fig, axs = plt.subplots(5, 6, figsize=(24, 16))
fig.suptitle('First 15 Images: Original (left) and Histogram Equalized (right)', fontsize=16)

# Iterate through the first 15 images
for i, img_file in enumerate(image_files):
    # Read the image using OpenCV in grayscale mode
    img = cv2.imread(os.path.join(image_path, img_file), cv2.IMREAD_GRAYSCALE)

    # Check if image is successfully loaded
    if img is None:
        print(f"Failed to load {img_file}")
        continue

    # Original image (grayscale)
    img_original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB for displaying with matplotlib

    # Apply histogram equalization to enhance contrast
    img_eq = cv2.equalizeHist(img)

    # Convert preprocessed image to RGB for displaying with matplotlib
    img_preprocessed = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)

    # Determine the row and column for the current image
    row = i // 3  # Display 3 sets of images per row
    col = (i % 3) * 2  # Each image set takes up two columns (original + preprocessed)

    # Display the original image in the grid
    axs[row, col].imshow(img_original)
    axs[row, col].axis('off')  # Hide axis
    axs[row, col].set_title(f"Original {i + 1}")

    # Display the preprocessed image next to the original
    axs[row, col + 1].imshow(img_preprocessed)
    axs[row, col + 1].axis('off')  # Hide axis
    axs[row, col + 1].set_title(f"Histogram Equalized {i + 1}")

# Adjust layout and display the grid
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
plt.show()
