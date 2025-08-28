# Handwritten Character Recognition

## Overview
This project is a web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Built with PyTorch and deployed via Streamlit, it allows users to upload a PNG/JPEG image of a handwritten digit and receive a prediction with confidence score. The app achieves ~98% accuracy on the MNIST test set.

## Files
- `app.py`: Streamlit web app for digit recognition.
- `digit_model.pth`: Pre-trained CNN model.
- `requirements.txt`: Python dependencies.
- `train_mnist_model.py`: Script to train the CNN on MNIST.
- `save_mnist_samples.py`: Generates sample MNIST test images.
- `digit.py`: [Optional; clarify purpose if included].

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Faiz1310/CodeAlpha_Projects.git
   cd CodeAlpha_Projects/HandwrittenCharacterRecognition
   ```
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure `digit_model.pth`**: Included in the repository. If missing, run `train_mnist_model.py` to generate it.

## Usage
1. **Run Locally**:
   ```bash
   streamlit run app.py
   ```
   - Opens at `http://localhost:8501`.
2. **Upload Image**:
   - Upload a PNG/JPEG of a handwritten digit (white background, dark ink).
   - Click "Predict Digit" to see the predicted digit and confidence score.
3. **Sample Inputs**:
   - Use `save_mnist_samples.py` to generate MNIST test images (`sample_digits/digit_X_Y.png`).
   - Draw digits in Paint (100x100, white background, black brush) and save as PNG.

## Deployment
- Deployed on Streamlit Sharing: [Insert deployed URL, e.g., https://faiz1310-codealpha-handwritten.streamlit.app].
- To deploy:
  1. Go to https://share.streamlit.io.
  2. Select `CodeAlpha_Projects` repo, set main file to `HandwrittenCharacterRecognition/app.py`.
  3. Deploy and access the public URL.

## Notes
- The CNN expects 28x28 grayscale images, matching MNIST format.
- For best results, ensure clear, centered digits in uploaded images.
- Part of the `CodeAlpha_Projects` portfolio, alongside Speech Emotion Recognition.

## Author
- Faiz (GitHub: [Faiz1310](https://github.com/Faiz1310))
