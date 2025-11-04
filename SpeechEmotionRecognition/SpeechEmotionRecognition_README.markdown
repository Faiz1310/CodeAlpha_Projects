# Speech Emotion Recognition

## Overview
This project is a web application that classifies emotions (e.g., Neutral, Happy, Sad, Angry) from speech audio files using a machine learning model trained on the RAVDESS dataset. Built with Python and Streamlit, users can upload a WAV audio file (2-5 seconds) to receive a predicted emotion with a confidence score. The model achieves approximately 80% accuracy on the test set.

## Files
- `app.py`: Streamlit web app for emotion recognition.
- `emotion_model.pth`: Pre-trained model.
- `requirements.txt`: Python dependencie].
- `train_emotion_model.py`: Script to train the model.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Faiz1310/CodeAlpha_Projects.git
   cd CodeAlpha_Projects/SpeechEmotionRecognition
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
   - Example dependencies: `streamlit`, `torch`, `torchaudio`, `librosa`, `numpy` [Update based on your `requirements.txt`].
4. **Download Dataset** (if training):
   - Get RAVDESS from https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio.
   - Unzip to `data/RAVDESS` [Update path if different].

## Usage
1. **Run Locally**:
   ```bash
   streamlit run app.py
   ```
   - Opens at `http://localhost:8501`.
2. **Upload Audio**:
   - Upload a WAV file (2-5 seconds, mono, 48kHz preferred).
   - Click "Predict Emotion" to view the predicted emotion and confidence score.
3. **Sample Inputs**:
   - Use RAVDESS audio files (e.g., `data/RAVDESS/Actor_01/03-01-03-01-01-01-01.wav` for Happy).
   - Record speech in Audacity, export as WAV (48kHz, mono).

## Deployment
- Deployed on Streamlit Sharing: [Insert deployed URL, e.g., https://faiz1310-codealpha-emotion.streamlit.app].
- To deploy:
  1. Go to https://share.streamlit.io.
  2. Select `CodeAlpha_Projects`, set main file to `SpeechEmotionRecognition/app.py`.
  3. Deploy and access the public URL.

## Notes
- Ensure audio files are clear and match the training format (e.g., WAV, 48kHz).
- Noisy or low-quality audio may reduce prediction accuracy.
- Part of the `CodeAlpha_Projects` portfolio, alongside Handwritten Character Recognition.

## Author
- Faiz (GitHub: [Faiz1310](https://github.com/Faiz1310))
