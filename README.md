# Bird Species Classification with Deep Learning

This project aims to classify bird species using deep learning techniques. The model is built using TensorFlow and Keras, leveraging the powerful InceptionV3 architecture for feature extraction. The dataset consists of images of different bird species, divided into training, validation, and evaluation sets.

## Dataset Description:
Drive Link for Dataset : https://drive.google.com/file/d/1-K7alGY1R9ntsGDCWsJTRN46ACW1vtto/view?usp=sharing
The dataset comprises images of various bird species, with corresponding labels:

- Crane
- Crow
- Egret
- Kingfisher
- Myna
- Peacock
- Pitta
- Rosefinch
- Tailorbird
- Wagtail

## Folder Structure:

- **Train_data**: Contains images used for training the model.
- **Validation_data**: Contains images used for validating the model during training.
- **Eval_data**: This folder is not directly used in the training process and is intended for evaluating the model after training.

## Model Architecture:

The model architecture is based on the InceptionV3 convolutional neural network. It is initialized with weights pre-trained on the ImageNet dataset and further fine-tuned for the bird species classification task.

## Training:

The model is trained using transfer learning, where the pre-trained InceptionV3 layers are frozen initially, and only the newly added layers are trained. This approach helps in efficient training with a smaller dataset.

## Evaluation Metrics:

During training, the model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into the model's ability to correctly classify bird species.

## Usage:

To train the model, ensure the dataset is organized as described above and execute the provided Python script. Adjust hyperparameters as needed for optimal performance.

## Dependencies:

- TensorFlow
- Keras
- scikit-learn

## Acknowledgments:

The implementation in this repository draws inspiration from various online resources and tutorials on deep learning and image classification.


