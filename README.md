# 🌱 AI-Based Plant Classification and Disease Detection

An AI-powered image classification project for identifying plant types and detecting their health conditions using Deep Learning.

## 🎯 Project Overview

This project uses image-based deep learning techniques to classify plant images and identify whether the plant is healthy or affected by a specific disease.

The project includes:
- 🌱 Plant type classification
- 🩺 Plant disease / health condition detection
- 🧠 CNN-based image classification
- 🤖 MLP model for comparison
- 📊 Model evaluation and visualization
- 🖥️ Streamlit graphical interface

## 🧠 Models

Two neural network approaches were implemented:

### CNN
A Convolutional Neural Network was developed for image classification using:
- Convolutional layers
- Max Pooling
- ReLU activation
- Dropout
- Softmax output layer

### MLP
A Multi-Layer Perceptron was implemented as a comparison model using flattened image features.

## 📊 Dataset

The project uses the **PlantVillage dataset**.

A balanced subset of the dataset was created for training and evaluation.

- Total images: 16,000
- Image size: 128 × 128
- Train/Test split: 80% / 20%

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook

## 📊 Jupyter Notebook

The notebook contains data analysis, preprocessing,
visualizations, model training and evaluation.

## 🎥 Application Demo

A short demo of the Streamlit application is available here:

[▶️ Watch the Demo](VIDEO_LINK)

## 📁 Project Structure


```text
plant-classification-ai/
│
├── app.py
├── train_model.py
├── plant_classification_analysis.ipynb
├── classes.json
├── requirements.txt
├── .gitignore
└── README.md
