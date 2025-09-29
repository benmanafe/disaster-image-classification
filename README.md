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

[cite_start]Natural disasters require fast and efficient identification to ensure safety[cite: 9]. [cite_start]Manually classifying disaster images is often slow and tedious[cite: 10, 11]. [cite_start]This project leverages Convolutional Neural Networks (CNNs) to automate the classification of disaster images into four categories: **Earthquake, Water Disasters, Landslides, and Urban Fire**[cite: 12, 14, 17].

We implement and evaluate two approaches:
1.  [cite_start]A **custom CNN model** built from the ground up[cite: 1, 52, 54].
2.  [cite_start]A **transfer learning approach** using the lightweight MobileNetV2 architecture[cite: 1, 53, 58].

---

## Dataset

[cite_start]The project uses a dataset containing **1946 images** of natural disasters[cite: 20, 36].

* [cite_start]**Classes:** Earthquake, Land Slide, Urban Fire, and Water Disaster[cite: 17, 36].
* [cite_start]**Imbalance:** The dataset is heavily imbalanced[cite: 20, 38]. [cite_start]The Water Disaster class has 1035 images, the Landslide and Urban Fire classes have 456 and 419 images respectively, and the Earthquake class has only 36 images[cite: 21, 37].
* [cite_start]**Image Size:** The original image sizes are inconsistent, ranging from 157x220 to 820x2048 pixels[cite: 23, 39].

---

## Methodology

### 1. Data Preparation
* [cite_start]**Splitting:** The dataset was split into training (70%), validation (10%), and testing (20%) sets[cite: 44].
* [cite_start]**Balancing:** To address the class imbalance, a combination of **oversampling** and **undersampling** was applied to the training set[cite: 67].
* [cite_start]**Augmentation & Resizing:** All images were resized to a standard 224x224[cite: 80, 81]. [cite_start]The training data was further augmented using `RandomHorizontalFlip`, `RandomRotation`, and `RandomResizedCrop` to improve model generalization[cite: 82].

### 2. Model Architectures
* [cite_start]**Scratch CNN:** A custom CNN was designed with four sequential convolutional blocks[cite: 105]. [cite_start]Each block consists of `Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d` layers[cite: 100, 101, 103, 104]. [cite_start]The number of output channels doubled with each block (16, 32, 64, 128)[cite: 105, 106]. [cite_start]A fully connected classifier was added at the end to output predictions for the four classes[cite: 110, 113].
* [cite_start]**MobileNetV2 (Transfer Learning):** The pre-trained MobileNetV2 model was used[cite: 147]. Two strategies were tested:
    1.  [cite_start]**Frozen:** All weights of the pre-trained layers were frozen, and only the custom classifier was trained[cite: 170].
    2.  [cite_start]**Unfrozen:** The weights of the entire network were fine-tuned on the disaster dataset[cite: 170].

### 3. Training
* [cite_start]**Optimizer:** Adam[cite: 119].
* [cite_start]**Loss Function:** CrossEntropyLoss, suitable for multiclass classification[cite: 118].
* [cite_start]**Epochs:** The scratch model was trained for 50 epochs, while both MobileNetV2 models were trained for 20 epochs[cite: 130, 173].
* [cite_start]**Learning Rates:** $1 \times 10^{-3}$ for the scratch model and $1 \times 10^{-5}$ for the MobileNetV2 models[cite: 119, 168].

---

## Results

[cite_start]The performance of the three models was evaluated on the unseen test set[cite: 57, 63]. [cite_start]The MobileNetV2 (Unfrozen) model achieved the best results across all metrics[cite: 181, 182, 225].

| Model | Loss | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Scratch Model | 1.232878 | 0.756410 | 0.781836 | 0.756410 | 0.757539 |
| MobileNetV2 (Frozen) | 0.587004 | 0.834135 | 0.829805 | 0.834135 | 0.819009 |
| **MobileNetV2 (Unfrozen)** | **0.275663** | **0.918269** | **0.915818** | **0.918269** | **0.912862** |
[cite_start]_Data sourced from the results table in the report[cite: 179]._

[cite_start]The unfrozen MobileNetV2 model's superior performance is likely due to its ability to adapt its pre-trained weights to the specific features of the disaster image dataset, resulting in better adaptation and higher accuracy[cite: 183, 184].

---

## Conclusion

[cite_start]Pre-trained models, specifically **MobileNetV2**, significantly outperform a CNN built from scratch for this disaster classification task[cite: 225]. [cite_start]Fine-tuning the entire network (the "unfrozen" approach) yields the best performance, achieving higher accuracy and better metrics than the frozen model and the scratch model[cite: 225]. This demonstrates the power of transfer learning for image classification problems.

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
