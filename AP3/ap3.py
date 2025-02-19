import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import norm

def calculate_error(original, resized):
    if original.ndim == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if resized.ndim == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return norm(original.astype(float) - resized.astype(float), 'fro')

def plot_images(original, resized, title_original, title_resized):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(title_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title(title_resized)
    plt.axis('off')

    plt.show()

image_path = "text.jpg" 
original_image = cv2.imread(image_path)

if original_image is None:
    print("Error: Image not found. Please provide a valid image path.")
    exit()

downscaled_image = cv2.resize(original_image, (original_image.shape[1] // 4, original_image.shape[0] // 4), interpolation=cv2.INTER_AREA)

nearest_neighbor_resized = cv2.resize(downscaled_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

bicubic_resized = cv2.resize(downscaled_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)

error_nearest = calculate_error(original_image, nearest_neighbor_resized)
error_bicubic = calculate_error(original_image, bicubic_resized)

plot_images(original_image, nearest_neighbor_resized, "Original Image", "Nearest Neighbor Resized")
plot_images(original_image, bicubic_resized, "Original Image", "Bicubic Resized")

print(f"Error (Frobenius norm) using Nearest Neighbor: {error_nearest:.2f}")
print(f"Error (Frobenius norm) using Bicubic: {error_bicubic:.2f}")

if error_nearest > error_bicubic:
    print("Bicubic interpolation preserves details better compared to Nearest Neighbor.")
else:
    print("Nearest Neighbor works better in this case.")

