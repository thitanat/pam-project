# Passive Acoustic Monitoring (PAM) Project

This project focuses on preprocessing audio data into mel spectrogram images and training two classifiers for species identification using Passive Acoustic Monitoring (PAM) techniques. The classifiers include a Convolutional Neural Network (CNN) built from scratch and a pretrained audio model fine-tuned on the BirdCLEF 2025 dataset.

## Project Structure

The project is organized into the following directories and files:

- **data/**: Contains raw and processed audio data.
  - **raw_audio/**: Original audio recordings used for the PAM project.
  - **processed/**: Holds processed data, including:
    - **mel_spectrograms/**: Generated mel spectrogram images from audio clips.
    - **labels/**: Encoded labels corresponding to the audio clips.
  - **train_metadata.csv**: Metadata for the training data, including filenames and species labels.

- **notebooks/**: Jupyter notebooks for data exploration, preprocessing, and model training.
  - **data_exploration.ipynb**: Explore the dataset, visualize audio samples, and understand species distribution.
  - **preprocessing.ipynb**: Guide through preprocessing steps, including loading audio files and generating mel spectrograms.
  - **model_training.ipynb**: Train models, evaluate performance, and visualize results.

- **scripts/**: Python scripts for preprocessing and training models.
  - **preprocess.py**: Handles preprocessing of audio files into mel spectrograms and saves them with labels.
  - **train_cnn.py**: Defines and trains a CNN from scratch using processed mel spectrograms.
  - **train_pretrained.py**: Fine-tunes a pretrained audio model on the BirdCLEF data.
  - **utils.py**: Contains utility functions for loading data, visualizing results, and calculating metrics.

- **models/**: Defines model architectures.
  - **cnn_model.py**: Architecture of the Convolutional Neural Network.
  - **pretrained_model.py**: Architecture and methods for loading and fine-tuning the pretrained model.

- **outputs/**: Stores logs, checkpoints, and results of the training.
  - **logs/**: Log files generated during training.
  - **checkpoints/**: Model checkpoints saved during training.
  - **results/**: Contains:
    - **accuracy_scores.txt**: Records accuracy scores of trained models.
    - **confusion_matrices/**: Stores confusion matrix visualizations for model evaluation.

- **requirements.txt**: Lists required Python packages and dependencies for the project.

- **README.md**: Overview of the project, setup instructions, and usage guidelines.

- **.gitignore**: Specifies files and directories to be ignored by version control.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pam-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your raw audio files in the `data/raw_audio/` directory.

4. Run the preprocessing notebook (`notebooks/preprocessing.ipynb`) to generate mel spectrograms and labels.

5. Use the model training notebook (`notebooks/model_training.ipynb`) to train the classifiers.

## Usage Guidelines

- Ensure that your audio files are in the correct format and placed in the `data/raw_audio/` directory.
- Follow the notebooks for step-by-step instructions on data exploration, preprocessing, and model training.
- Check the `outputs/` directory for logs, checkpoints, and results after training.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.# pam-project
