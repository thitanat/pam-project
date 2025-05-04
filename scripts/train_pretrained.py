# filepath: /pam-project/pam-project/scripts/train_pretrained.py

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from models.pretrained_model import load_pretrained_model  # Assuming this function is defined in pretrained_model.py

# Set parameters
sample_rate = 22050
clip_duration = 5
samples_per_clip = sample_rate * clip_duration
mel_spectrogram_dir = "../data/processed/mel_spectrograms/"
labels_dir = "../data/processed/labels/"
metadata_file = "../data/train_metadata.csv"

# Load metadata
metadata = pd.read_csv(metadata_file)

# Initialize lists to hold data and labels
X = []  # will hold mel spectrogram arrays
y = []  # will hold corresponding species labels

# Load mel spectrograms and labels
for idx, row in metadata.iterrows():
    mel_spec_path = os.path.join(mel_spectrogram_dir, f"{row['filename']}.npy")
    if os.path.exists(mel_spec_path):
        mel_spec = np.load(mel_spec_path)
        X.append(mel_spec)
        y.append(row['species'])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Load the pretrained model
model = load_pretrained_model(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_encoded)))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save("../outputs/checkpoints/pretrained_model.h5")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("../outputs/results/model_accuracy.png")
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("../outputs/results/model_loss.png")
plt.close()

# Save accuracy scores to a text file
with open("../outputs/results/accuracy_scores.txt", "w") as f:
    f.write(f"Training Accuracy: {history.history['accuracy'][-1]}\n")
    f.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]}\n")