import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Define CNN model (same as training)
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
@st.cache_resource
def load_model():
    model = DigitCNN()
    try:
        model.load_state_dict(torch.load('digit_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'digit_model.pth' not found. Please train the model first.")
        st.stop()

model = load_model()

# Streamlit app
st.title("Handwritten Digit Recognition")
st.markdown("""
Upload an image of a handwritten digit (0-9) on a white background with dark ink.
Image will be resized to 28x28 pixels. Trained on MNIST dataset.
""")

# Image upload
uploaded_file = st.file_uploader("Choose an image (PNG/JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process button
    if st.button("Predict Digit"):
        try:
            # Preprocess image
            image = image.convert('L')  # Grayscale
            image = image.resize((28, 28))  # Resize to 28x28
            image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1], ensure float32
            # Invert if dark background (MNIST has white digits on black)
            if image_array.mean() > 0.5:
                image_array = 1 - image_array
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            image_tensor = transform(image_array).unsqueeze(0)  # [1, 1, 28, 28]
            image_tensor = image_tensor.float()  # Ensure float32

            # Predict
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_digit = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_digit].item() * 100

            # Display result
            st.success(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}. Please upload a valid PNG/JPEG image of a single digit.")

# Footer
st.markdown("---")
st.markdown(" Upload a digit image to predict! [GitHub Repo](https://github.com/Faiz/handwritten-character-recognition)")