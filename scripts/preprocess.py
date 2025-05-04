# filepath: /pam-project/pam-project/scripts/preprocess.py
# Import necessary libraries for audio and data processing
import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for audio processing
sample_rate = 22050        # Audio sampling rate (Hz)
clip_duration = 5          # Duration of each clip in seconds
samples_per_clip = sample_rate * clip_duration

# Path to audio data and metadata
audio_dir = "../data/raw_audio/"         # directory containing audio files
metadata_file = "../data/train_metadata.csv"  # CSV with columns ["filename","species"]
mel_spec_dir = "../data/processed/mel_spectrograms/"  # directory to save mel spectrograms
label_dir = "../data/processed/labels/"  # directory to save labels

# Create directories if they do not exist
os.makedirs(mel_spec_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_file)  # expects columns 'filename' and 'species'

# Initialize lists to hold data and labels
X = []   # will hold mel spectrogram arrays
y = []   # will hold corresponding species labels

# Process each audio file
for idx, row in metadata.iterrows():
    file_path = os.path.join(audio_dir, row['filename'])
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping.")
        continue  # skip missing files
    
    # Load the full audio file
    signal, sr = librosa.load(file_path, sr=sample_rate)
    
    # Split into non-overlapping clips of fixed duration
    for start in range(0, len(signal), samples_per_clip):
        clip = signal[start:start + samples_per_clip]
        
        # If clip is too short, skip it
        if len(clip) < samples_per_clip:
            continue
        
        # Compute the mel spectrogram (128 mel bands)
        mel_spec = librosa.feature.melspectrogram(
            y=clip, sr=sr, n_mels=128, hop_length=512
        )
        
        # Convert to log scale (decibels) for better representation
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save mel spectrogram as an image
        mel_spec_image_path = os.path.join(mel_spec_dir, f"{row['species']}_{idx}_{start//samples_per_clip}.png")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram of {row["species"]}')
        plt.savefig(mel_spec_image_path)
        plt.close()
        
        # Append to our dataset
        X.append(mel_spec_db)
        y.append(row['species'])

# Save labels to a text file
labels_path = os.path.join(label_dir, "labels.txt")
with open(labels_path, 'w') as f:
    for label in y:
        f.write(f"{label}\n")

print("Preprocessing complete. Mel spectrograms and labels have been saved.")