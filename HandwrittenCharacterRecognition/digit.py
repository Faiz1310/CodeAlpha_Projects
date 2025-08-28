import numpy as np
from sklearn.datasets import load_digits
from PIL import Image
import os

# Load digits dataset
digits = load_digits()

# Create output folder if not exists
output_folder = "digit_samples"
os.makedirs(output_folder, exist_ok=True)

# For digits 0-9, find first occurrence and save as 28x28 PNG
for digit in range(10):
    # Find first image of the digit
    index = np.where(digits.target == digit)[0][0]
    img_8x8 = digits.images[index]

    # Scale pixel values (0-16) to (0-255)
    img_scaled = (img_8x8 * 16).astype(np.uint8)

    # Resize to 28x28 (MNIST format)
    img_resized = Image.fromarray(img_scaled).resize((28, 28), Image.LANCZOS)

    # Save the image
    filename = os.path.join(output_folder, f"digit_{digit}.png")
    img_resized.save(filename)
    print(f"Saved digit {digit} as {filename}")
