# 🐾 Cats vs. Dogs Image Classification

This project is focused on building a Convolutional Neural Network (CNN) to classify images of cats and dogs using the popular [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle.

## 🚀 Project Overview

The goal of this project is to develop a deep learning model that can accurately differentiate between images of cats and dogs. Through this project, I gained hands-on experience with image data preprocessing, model building, and performance optimization.

## 📁 Dataset

The dataset contains 25,000 labeled images of cats and dogs (12,500 images of each). The images are split into training and validation sets:
- **Training Set:** 20,000 images
- **Validation Set:** 5,000 images

## 🛠️ Tools and Technologies

- **Python**
- **TensorFlow & Keras:** For building and training the CNN model.
- **OpenCV:** For image processing.
- **Matplotlib:** For visualizing model performance.

## 🧠 Model Architecture

The CNN model was designed with the following layers:
- **Convolutional Layers:** To extract features from images.
- **MaxPooling Layers:** To reduce the spatial dimensions.
- **Batch Normalization:** To stabilize and accelerate training.
- **Dropout Layers:** To prevent overfitting.
- **Fully Connected Layers:** To make predictions based on the extracted features.

## ⚙️ Training the Model

The model was trained on the training dataset with a batch size of 32 over 10 epochs. The loss function used was `binary_crossentropy`, and the optimizer was `adam`.

### Performance Metrics

- **Training Accuracy:** ~93%
- **Validation Accuracy:** ~81%

## 📊 Results

The model performs well on both the training and validation sets, though there is some overfitting which can be addressed in future iterations by incorporating techniques like data augmentation.

## 🔍 Visualizing Performance

The training and validation accuracy and loss were plotted to monitor the model's performance over the epochs.

![Model Accuracy]
![Model Loss]

## 🔄 Next Steps

To further improve the model, the following steps can be taken:
- **Data Augmentation:** To increase the diversity of the training set.
- **Hyperparameter Tuning:** To find the optimal model settings.
- **Experimenting with Different Architectures:** To see if more complex models perform better.

## 📂 Project Structure

- `train/`: Directory containing training images.
- `test/`: Directory containing validation images.
- `model.py`: Python script for building and training the model.
- `plots/`: Directory containing accuracy and loss plots.

## 🤝 Contributing

If you have suggestions or improvements, feel free to fork this repository and submit a pull request.
