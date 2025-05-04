# filepath: /pam-project/pam-project/scripts/train_cnn.py
# This script defines and trains a Convolutional Neural Network from scratch using the processed mel spectrograms.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set parameters
sample_rate = 22050
clip_duration = 5
samples_per_clip = sample_rate * clip_duration
mel_spectrogram_dir = "../data/processed/mel_spectrograms/"
labels_dir = "../data/processed/labels/"
metadata_file = "../data/train_metadata.csv"

# Load metadata
metadata = pd.read_csv(metadata_file)

# Load mel spectrograms and labels
X = []
y = []

for idx, row in metadata.iterrows():
    mel_spec_path = os.path.join(mel_spectrogram_dir, row['filename'].replace('.wav', '.npy'))
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

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Create and compile the model
input_shape = (X.shape[1], X.shape[2], 1)  # Assuming mel spectrograms are 2D
num_classes = len(np.unique(y_encoded))
model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape X for the model
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Save the model
model.save('../outputs/checkpoints/cnn_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig('../outputs/results/training_accuracy.png')
plt.show()