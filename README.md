# Image Captioning with CNN and RNN

This Jupyter notebook (`image-captioning.ipynb`) implements an image captioning model using a combination of Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN). The model is trained on the Flickr8k dataset to generate descriptive captions for images.

## Overview

- **Dataset:** The Flickr8k dataset contains images with corresponding captions. The notebook guides you through the process of downloading and setting up the dataset for training.

- **Model Architecture:**
  - **Encoder CNN (InceptionV3):** Pretrained on ImageNet, the CNN extracts meaningful features from input images.
  - **Decoder RNN (LSTM):** Generates captions based on the features obtained from the CNN.

- **Training:**
  - The notebook provides code to load the dataset, build the model, and train it over a specified number of epochs.
  - Hyperparameters such as embedding size, hidden size, learning rate, and more are configurable for experimentation.

- **Evaluation:**
  - The model's captions are evaluated on randomly selected images from the dataset.
  - Evaluation includes displaying actual and predicted captions, as well as calculating the similarity between them using Universal Sentence Encoder.

- **Results:**
  - Evaluation results, including actual and predicted captions along with their similarity scores, are presented in a clear format for analysis.

## Usage

1. **Dataset Setup:**
   - Follow the instructions in the notebook to download and organize the Flickr8k dataset.

2. **Model Training:**
   - Execute the code cells related to model training, adjusting hyperparameters if desired.

3. **Evaluation:**
   - Run the evaluation section to assess the model's performance on a set of randomly selected images.
   - View actual and predicted captions, along with similarity scores.

4. **Experimentation:**
   - Feel free to experiment with hyperparameters, model architecture, or dataset variations to observe different outcomes.

## Requirements

- Python 3
- Required libraries (PyTorch, torchvision, pandas, spacy, tqdm, torchtext, PIL, matplotlib, scikit-learn, tensorflow, tensorflow-hub)

