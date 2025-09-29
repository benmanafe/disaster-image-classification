# Disaster Image Classification: MobileNetV2 vs. Scratch CNN

This project explores the classification of natural disaster images using deep learning. It compares the performance of a Convolutional Neural Network (CNN) built from scratch against a pre-trained MobileNetV2 model on a multiclass disaster dataset. An interactive web app is also included for live classification.

## Table of Contents
* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Methodology](#methodology)
* [Results](#results)
* [Conclusion](#conclusion)
* [Interactive Demo App](#interactive-demo-app)
---

## Project Overview

Natural disasters require fast and efficient identification to ensure safety. Manually classifying disaster images is often slow and tedious. This project leverages Convolutional Neural Networks (CNNs) to automate the classification of disaster images into four categories: **Earthquake, Water Disasters, Landslides, and Urban Fire**.

We implement and evaluate two approaches:
1.  A **custom CNN model** built from the ground up.
2.  A **transfer learning approach** using the lightweight MobileNetV2 architecture.

---

## Dataset

The project uses a dataset containing **1946 images** of natural disasters.

* **Classes:** Earthquake, Land Slide, Urban Fire, and Water Disaster.
* **Imbalance:** The dataset is heavily imbalanced. The Water Disaster class has 1035 images, the Landslide and Urban Fire classes have 456 and 419 images respectively, and the Earthquake class has only 36 images.
* **Image Size:** The original image sizes are inconsistent, ranging from 157x220 to 820x2048 pixels.
* **Source:** [Disaster Images Dataset on Kaggle](https://www.kaggle.com/datasets/varpit94/disaster-images-dataset)

---

## Methodology

### 1. Data Preparation
* **Splitting:** The dataset was split into training (70%), validation (10%), and testing (20%) sets.
* **Balancing:** To address the class imbalance, a combination of **oversampling** and **undersampling** was applied to the training set.
* **Augmentation & Resizing:** All images were resized to a standard 224x224. The training data was further augmented using `RandomHorizontalFlip`, `RandomRotation`, and `RandomResizedCrop` to improve model generalization.

### 2. Model Architectures
* **Scratch CNN:** A custom CNN was designed with four sequential convolutional blocks. Each block consists of `Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d` layers. The number of output channels doubled with each block (16, 32, 64, 128). A fully connected classifier was added at the end to output predictions for the four classes.
* **MobileNetV2 (Transfer Learning):** The pre-trained MobileNetV2 model was used. Two strategies were tested:
    1.  **Frozen:** All weights of the pre-trained layers were frozen, and only the custom classifier was trained.
    2.  **Unfrozen:** The weights of the entire network were fine-tuned on the disaster dataset.

### 3. Training
* **Optimizer:** Adam.
* **Loss Function:** CrossEntropyLoss, suitable for multiclass classification.
* **Epochs:** The scratch model was trained for 50 epochs, while both MobileNetV2 models were trained for 20 epochs.
* **Learning Rates:** $1 \times 10^{-3}$ for the scratch model and $1 \times 10^{-5}$ for the MobileNetV2 models.

---

## Results

The performance of the three models was evaluated on the unseen test set. The MobileNetV2 (Unfrozen) model achieved the best results across all metrics.

| Model | Loss | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Scratch Model | 1.232878 | 0.756410 | 0.781836 | 0.756410 | 0.757539 |
| MobileNetV2 (Frozen) | 0.587004 | 0.834135 | 0.829805 | 0.834135 | 0.819009 |
| **MobileNetV2 (Unfrozen)** | **0.275663** | **0.918269** | **0.915818** | **0.918269** | **0.912862** |

_Data sourced from the results table in the report._

The unfrozen MobileNetV2 model's superior performance is likely due to its ability to adapt its pre-trained weights to the specific features of the disaster image dataset, resulting in better adaptation and higher accuracy.

---

## Conclusion

Pre-trained models, specifically **MobileNetV2**, significantly outperform a CNN built from scratch for this disaster classification task. Fine-tuning the entire network (the "unfrozen" approach) yields the best performance, achieving higher accuracy and better metrics than the frozen model and the scratch model. This demonstrates the power of transfer learning for image classification problems.

---

## Interactive Demo App

This project includes an interactive web application built with **Streamlit** that allows you to classify your own disaster images using the trained MobileNetV2 model.

### Features
* **Image Uploader:** Supports various image formats (`jpg`, `png`, `webp`, etc.).
* **Real-time Classification:** Get instant predictions with a click of a button.
* **Confidence Score:** View the model's confidence in its prediction.
* **Confidence Threshold:** Use the sidebar slider to set a threshold, below which predictions are flagged as "low confidence."
* **Full Probability Distribution:** See the model's predicted probability for all four disaster classes.



### Live Demo
You can try the live application without any local setup. Click the link below to access the deployed Streamlit app:

**[➡️ Access the Disaster Classification App Here](https://disaster-app-classificationpy-uujevwjsxu8gqgmrjzgdot.streamlit.app/)**
