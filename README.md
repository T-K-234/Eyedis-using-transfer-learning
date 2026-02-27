
# AI-Based Eye Disease Classification System with Report Summarization

## Project Overview

This project presents a **Flask-based web application for automated eye disease classification using retinal images**. The system uses **deep learning and ensemble learning techniques** to classify eye diseases and generate a summarized diagnostic report for users.

The application allows users to **upload retinal images through a web interface**, after which the trained deep learning models analyze the image and predict the disease category.

The system supports **multiclass classification** and can identify multiple eye diseases such as **Diabetic Retinopathy, Glaucoma, Cataract, and Healthy Eyes**.

Additionally, the application generates a **short medical report summarizing the prediction and basic disease information**, making it easier for users to understand the results.

---

# Introduction

Eye diseases are one of the major causes of vision impairment worldwide. Early diagnosis plays a crucial role in preventing severe vision loss. However, traditional diagnosis requires expert ophthalmologists and specialized medical equipment.

With advancements in **deep learning and computer vision**, retinal image analysis can be automated to assist medical professionals in diagnosing diseases quickly and accurately.

This project integrates **multiple deep learning models through ensemble learning** to improve classification performance. The final system is deployed as a **Flask web application**, making it accessible and easy to use.

---

# Key Features

• Multiclass eye disease classification
• Ensemble learning for improved accuracy
• Upload retinal images through a web interface
• Automatic prediction of eye disease type
• Report summarization explaining the prediction
• Flask-based backend for model inference
• Simple and user-friendly interface

---

# Dataset

The model was trained on a dataset of retinal fundus images collected from Kaggle.

### Classes

The dataset contains four classes:

* Healthy
* Diabetic Retinopathy
* Glaucoma
* Cataract

### Dataset Size

| Dataset  | Images |
| -------- | ------ |
| Training | 3,372  |
| Testing  | 845    |

---

# Data Preprocessing

Before training the models, the retinal images were preprocessed to improve model performance.

### Preprocessing Techniques

• Image resizing to fixed dimensions
• Pixel normalization
• Data augmentation (rotation, flipping, zooming)
• Noise reduction and image enhancement

These preprocessing steps help the model learn better features from the retinal images.

---

# Deep Learning Models

The system uses **multiple deep learning models**, whose predictions are combined using **ensemble learning**.

## 1. Convolutional Neural Network (CNN)

A custom CNN architecture was used to learn spatial features from retinal images.

Key characteristics:

• Multiple convolution layers
• ReLU activation functions
• Max pooling layers
• Fully connected classification layer

CNN helps capture **local patterns such as lesions, blood vessels, and abnormal structures**.

---

## 2. ResNet-18

ResNet-18 is a deep convolutional neural network that uses **residual connections** to prevent vanishing gradients and allow deeper architectures.

Key characteristics:

• 18-layer deep architecture
• Residual skip connections
• Transfer learning using ImageNet weights

ResNet helps in learning **complex hierarchical features from retinal images**.

---

## 3. Swin Transformer

Swin Transformer is a **Vision Transformer architecture** designed for image classification tasks.

Key characteristics:

• Shifted window self-attention
• Hierarchical feature representation
• Efficient computation for high-resolution images

It captures **global relationships between image patches**, which improves classification performance.

---

# Ensemble Learning

Instead of relying on a single model, this project uses **ensemble learning**.

### How Ensemble Learning Works

1. Each model (CNN, ResNet-18, Swin Transformer) makes its own prediction.
2. The predictions are combined using **majority voting or probability averaging**.
3. The final predicted class is selected based on the combined output.

### Advantages of Ensemble Learning

• Higher accuracy
• Better generalization
• Reduced overfitting
• More robust predictions

---

# Web Application (Flask Integration)

The trained models are integrated into a **Flask web application** that allows users to interact with the system easily.

### Workflow

1. User uploads a retinal image through the web interface.
2. The image is preprocessed.
3. The ensemble model predicts the disease class.
4. The system generates a **summary report** explaining the prediction.
5. The result is displayed on the webpage.

---

# Report Summarization

After classification, the system generates a **short textual summary describing the detected disease**.

Example Output:

Prediction: **Diabetic Retinopathy**

Summary:
Diabetic Retinopathy is a complication caused by diabetes that affects the blood vessels of the retina. Early detection and regular monitoring are essential to prevent vision loss. It is recommended to consult an ophthalmologist for further evaluation.

This feature helps **users better understand the prediction results**.

---

# Training Configuration

| Parameter           | Value         |
| ------------------- | ------------- |
| Optimizer           | Adam          |
| Learning Rate       | 0.001         |
| Batch Size          | 32            |
| Loss Function       | Cross Entropy |
| Classification Type | Multiclass    |

---

# Applications

• Early screening of eye diseases
• AI-assisted medical diagnosis
• Telemedicine and remote healthcare
• Clinical decision support systems
• Automated retinal image analysis

---

# Future Improvements

• Deploy the system on cloud platforms
• Add more eye disease categories
• Improve model accuracy with larger datasets
• Integrate real-time hospital data systems
• Build a mobile application for disease detection

---

# Technologies Used

• Python
• PyTorch
• Tensorflow
• Flask
• OpenCV
• NumPy
• Matplotlib

---

# Author

**Tharun Kumar**
