# Eyedis-using-transfer-learning
# Eye Disease Detection Model using Retinal Images
# Project Overview
This repository contains an eye disease detection model built using ResNet-18 and VGGNet architectures. The models are designed to classify retinal images to detect various eye diseases. This project aims to assist in the early diagnosis and treatment of eye conditions by leveraging deep learning techniques.


# Introduction
The early detection of eye diseases such as diabetic retinopathy, glaucoma, and macular degeneration is crucial for preventing vision loss. This project uses convolutional neural networks (CNNs) to classify retinal images and identify signs of these conditions. The models are trained on a dataset of retinal images to achieve high accuracy in disease detection.

# Dataset
The models are trained on a dataset of retinal images, which includes healthy eyes and eyes with various diseases. The dataset is pre-processed to enhance image quality and normalize the data for optimal model performance.

Source: From kaggle
Classes: Healthy, Diabetic Retinopathy, Glaucoma,cataract.
Number of Images: train:3,372 Files(4 classes) ,test:845 Files(4 classes)
# Model Architectures
# ResNet-18
ResNet-18 is a convolutional neural network that employs residual learning to enable the training of very deep networks. It uses shortcut connections to mitigate the vanishing gradient problem, which is common in deep neural networks.

Layers: 18
Features: Uses skip connections to preserve gradient flow
Training: Fine-tuned with a learning rate of 0.001
# VGGNet
VGGNet is a deep convolutional network architecture known for its simplicity and depth. It uses very small convolution filters of size 3x3, which enables it to capture intricate features in the retinal images.

Layers: 16 (VGG-16) or 19 (VGG-19)
Features: Deep network with small filters
Training: Fine-tuned with a learning rate of 0.001
Training Process
The models were trained using the following steps:

# Data Preprocessing: 
Images were resized to a consistent resolution, normalized, and augmented to increase the diversity of the training data.
# Model Initialization:
Pre-trained weights on ImageNet were used to initialize the models, allowing them to benefit from prior learning.
# Training: 
The models were trained on a GPU using Adam optimizer, with a batch size of 32 and a learning rate of 0.001.
# Testing: 
The models were evaluated on a separate test set to assess their performance.
# Requirements
Python 3.8+
PyTorch
torchvision
NumPy
OpenCV
