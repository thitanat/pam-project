# utils.py

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(data_dir):
    """
    Load mel spectrogram images and corresponding labels from the specified directory.
    
    Parameters:
    data_dir (str): Directory containing mel spectrogram images and labels.
    
    Returns:
    X (list): List of mel spectrogram images.
    y (list): List of corresponding labels.
    """
    X = []
    y = []
    
    # Load images and labels
    mel_dir = os.path.join(data_dir, 'mel_spectrograms')
    label_dir = os.path.join(data_dir, 'labels')
    
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            label = f.read().strip()
        
        mel_file = label_file.replace('.txt', '.png')  # Assuming images are saved as PNG
        mel_path = os.path.join(mel_dir, mel_file)
        
        if os.path.exists(mel_path):
            mel_image = plt.imread(mel_path)
            X.append(mel_image)
            y.append(label)
    
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot the confusion matrix using seaborn heatmap.
    
    Parameters:
    y_true (list): True labels.
    y_pred (list): Predicted labels.
    class_names (list): List of class names for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy of predictions.
    
    Parameters:
    y_true (list): True labels.
    y_pred (list): Predicted labels.
    
    Returns:
    float: Accuracy score.
    """
    return accuracy_score(y_true, y_pred)