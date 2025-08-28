import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Emotion mapping from RAVDESS file codes (subset: neutral, happy, sad, angry)
emotion_map = {1: 'Neutral', 3: 'Happy', 4: 'Sad', 5: 'Angry'}  # Extend if needed
emotions = list(emotion_map.keys())  # [1, 3, 4, 5]

# Custom Dataset class
class SpeechDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            audio, sr = librosa.load(file_path, sr=48000)  # Load audio
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract 40 MFCCs
            # Handle empty or short audio
            if mfcc.shape[1] == 0:
                mfcc = np.zeros((40, 216))
            else:
                # Pad or trim to fixed length (216 frames)
                mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 216 - mfcc.shape[1]))), mode='constant')[:, :216]
            # Transpose to [seq_len, features] = [216, 40]
            mfcc = mfcc.T
            label = self.labels[idx]
            return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data to avoid crashing
            return torch.zeros(216, 40, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Load dataset paths and labels
def load_data(data_dir='data/RAVDESS'):
    file_paths = []
    labels = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} not found!")
    for actor_dir in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_dir)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    parts = file.split('-')
                    try:
                        emotion = int(parts[2])
                        if emotion in emotions:  # Filter to subset
                            file_paths.append(os.path.join(actor_path, file))
                            labels.append(emotions.index(emotion))  # Map to 0,1,2,3
                    except IndexError:
                        print(f"Skipping invalid file: {file}")
    return file_paths, labels

# LSTM Model
class EmotionLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=4):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input x: [batch_size, seq_len, input_size] = [batch_size, 216, 40]
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Main training
if __name__ == "__main__":
    # Load data
    file_paths, labels = load_data()
    print(f"Loaded {len(file_paths)} files.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

    train_dataset = SpeechDataset(X_train, y_train)
    test_dataset = SpeechDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced batch size
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionLSTM(num_classes=len(emotions)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(50):
        model.train()
        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save model
    torch.save(model.state_dict(), 'emotion_model.pth')
    print("Model saved as 'emotion_model.pth'")