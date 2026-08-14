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

The notebook contains data analysis, preprocessing, visualization, model training, and evaluation.

### Model Results

![Plant Detection Samples](screenshots/Plant%20detection%20samples.png)

![CNN Model Accuracy](screenshots/CNN%20model%20accuracy.png)

![MLP Model Accuracy](screenshots/MLP%20model%20accuracy.png)

![CNN and MLP Accuracy Comparison](screenshots/Accuracy%20comparison%20CNN%20%26%20MLP.png)

![Model Accuracy Comparison](screenshots/Comparison%20of%20Models%20Accuracy.png)

## 🖥️ Streamlit Application

The project also includes a graphical interface built with Streamlit for uploading plant images and displaying the model's prediction.

### Application Screenshots

![Application Interface](app-screenshots/01_app_interface.png)

![Image Upload](app-screenshots/02_image_upload.png)

![Prediction Process](app-screenshots/03_prediction_process.png)

![Prediction Result](app-screenshots/04_prediction_result.png)

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
├── README.md
│
├── screenshots/
│   ├── Accuracy comparison CNN & MLP.png
│   ├── CNN model accuracy.png
│   ├── Comparison of Models Accuracy.png
│   ├── MLP model accuracy.png
│   └── Plant detection samples.png
│
└── app-screenshots/
    ├── 01_app_interface.png
    ├── 02_image_upload.png
    ├── 03_prediction_process.png
    └── 04_prediction_result.png
