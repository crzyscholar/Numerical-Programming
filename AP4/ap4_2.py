import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# Puzzle parameters
rows = 4  # Number of rows
cols = 4  # Number of columns

def draw_horizontal_lines(ax, num_rows, img_width, img_height):
    for i in range(1, num_rows):
        y = i * (img_height // num_rows)
        ax.plot([0, img_width], [y, y], color='black', linewidth=1)

def draw_vertical_lines(ax, num_cols, img_width, img_height):
    for j in range(1, num_cols):
        x = j * (img_width // num_cols)
        ax.plot([x, x], [0, img_height], color='black', linewidth=1)

def draw_random_curves(ax, num_rows, num_cols, img_width, img_height):
    for i in range(1, num_rows):
        y = i * (img_height // num_rows)
        x_vals = np.linspace(0, img_width, 1000)
        freq = random.uniform(1, 3)
        amplitude = random.uniform(10, 50)  
        curve = y + amplitude * np.sin(freq * 2 * np.pi * x_vals / img_width)
        ax.plot(x_vals, curve, color='black', linewidth=1)

    for j in range(1, num_cols):
        x = j * (img_width // num_cols)
        y_vals = np.linspace(0, img_height, 1000)
        freq = random.uniform(1, 3) 
        amplitude = random.uniform(10, 50) 
        curve = x + amplitude * np.sin(freq * 2 * np.pi * y_vals / img_height)
        ax.plot(curve, y_vals, color='black', linewidth=1)

image_path = input("Enter the image filename (e.g., image.jpg): ") 
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Please provide a valid image filename.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_height, img_width, _ = image.shape

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, img_width)
ax.set_ylim(0, img_height)
ax.invert_yaxis()  
ax.axis('off')

ax.imshow(image_rgb)

draw_horizontal_lines(ax, rows, img_width, img_height)
draw_vertical_lines(ax, cols, img_width, img_height)
draw_random_curves(ax, rows, cols, img_width, img_height)

plt.show()

