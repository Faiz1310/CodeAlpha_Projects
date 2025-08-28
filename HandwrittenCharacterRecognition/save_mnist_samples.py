import torch
from torchvision import datasets, transforms
from PIL import Image
import os

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
os.makedirs('sample_digits', exist_ok=True)
for i in range(5):
    image, label = test_dataset[i]
    image = image.squeeze().numpy() * 255
    image = Image.fromarray(image.astype('uint8'), mode='L')
    image.save(f'sample_digits/digit_{label}_{i}.png')
    print(f"Saved sample_digits/digit_{label}_{i}.png")