# 🧠 Facial Expression Recognition using CNN (FER-2013)

This project implements a **Convolutional Neural Network (CNN)** for **facial expression recognition** using the **FER-2013 dataset**.  
The model classifies human facial images into **seven emotion categories** and is deployed through a **Streamlit web application** for real-time inference.

---

## 📌 Project Overview

Facial expression recognition is an important problem in **computer vision and affective computing**.  
This project focuses on building, training, evaluating, and deploying a deep learning model capable of identifying facial emotions from grayscale images.

**Emotion classes:**
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 📊 Dataset

- **Dataset:** FER-2013
- **Source:** Kaggle
- **Link: https:**//www.kaggle.com/datasets
- **Image size:** 48 × 48 pixels
- **Image type:** Grayscale
- **Classes:** 7
- **Structure:** Pre-split into `train/` and `test/` directories
- **Key challenge:** Class imbalance across emotions

The dataset was processed and trained using **Google Colab with GPU acceleration** due to its large size.

---

## 🏗️ Model Architecture

The model is a **deep Convolutional Neural Network (CNN)** designed to extract hierarchical facial features.

### Architecture Summary
- Convolutional layers with **ReLU activation**
- **MaxPooling** layers for spatial downsampling
- **Batch Normalization** for stable training
- **Dropout** layers to reduce overfitting
- Fully connected dense layer
- **Softmax output layer** for multi-class classification

**Input shape:** `(48, 48, 1)`  
**Output:** Probability distribution over 7 emotion classes

---

## ⚙️ Training Configuration

- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy, Precision, Recall, F1-score
- **Regularization:** Dropout + Batch Normalization
- **Early Stopping:** Enabled to prevent overfitting
- **Hardware:** Google Colab (GPU)

Training was automatically stopped when validation loss stopped improving.

---

## 📈 Evaluation

Model performance was evaluated using:
- Test accuracy
- Confusion matrix
- Classification report (Precision, Recall, F1-score)

The analysis shows that some emotions (e.g., *fear* and *surprise*) are more difficult to distinguish due to similar facial features.

---

## 🚀 Deployment (Streamlit App)

A **Streamlit-based web application** was developed to demonstrate the trained model.

### Features
- Upload a facial image
- Automatic preprocessing
- Emotion prediction with confidence scores
- Class probability breakdown

---

## 📂 Project Structure
CNN-Expression_classifier/
- ├── app.py
- ├── Facial_Expression_Recognition
- ├── fer2013_cnn_model.h5
- ├── requirements.txt
- └── README.md


---

## ▶️ How to Run the App Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/CNN-Expression_classifier.git
cd CNN-Expression_classifier


