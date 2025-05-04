# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the Convolutional Neural Network (CNN) architecture
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output
    model.add(layers.Flatten())
    
    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Example usage
if __name__ == "__main__":
    input_shape = (128, 128, 1)  # Example input shape for mel spectrograms
    num_classes = 10  # Example number of classes (species)
    
    cnn_model = create_cnn_model(input_shape, num_classes)
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Print the model summary
    cnn_model.summary()