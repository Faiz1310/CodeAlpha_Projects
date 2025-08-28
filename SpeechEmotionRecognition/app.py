import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import io
import soundfile as sf

# Emotion mapping (same as training)
emotion_map = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry'}

# LSTM Model (same as training)
class EmotionLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=4):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Load model
@st.cache_resource
def load_model():
    model = EmotionLSTM(num_classes=len(emotion_map))
    try:
        model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'emotion_model.pth' not found. Please train the model first.")
        st.stop()

model = load_model()

# Streamlit app
st.title("Speech Emotion Recognition")
st.markdown("""
Upload a short speech audio file (.wav) to predict the emotion.
Trained on RAVDESS dataset for Neutral, Happy, Sad, and Angry emotions.
""")

# Audio upload
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Play uploaded audio
    st.audio(uploaded_file)

    # Predict button
    if st.button("Predict Emotion"):
        try:
            # Read audio
            audio_bytes = uploaded_file.read()
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio.shape) > 1:  # Convert stereo to mono
                audio = np.mean(audio, axis=1)
            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            if mfcc.shape[1] == 0:
                st.error("Audio is too short or invalid!")
                st.stop()
            # Pad/trim to 216 frames
            mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 216 - mfcc.shape[1]))), mode='constant')[:, :216]
            # Transpose to [seq_len, features] = [216, 40]
            mfcc = mfcc.T
            # Create tensor with batch dimension: [1, 216, 40]
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

            # Predict
            with torch.no_grad():
                output = model(mfcc_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item() * 100

            # Display result
            emotion = emotion_map[predicted_idx]
            st.success(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}. Please upload a valid .wav file (2-5 seconds, mono, 48kHz).")

# Footer
st.markdown("---")
st.markdown(" Upload speech audio to predict emotion! [GitHub Repo](https://github.com/your-repo)")