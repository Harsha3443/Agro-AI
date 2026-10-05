# 🌱 AgroAI — Crop Disease Detection

AgroAI is a mobile application designed to detect plant diseases from leaf images using a deep learning model. The application uses a **MobileNetV2-based convolutional neural network** for image classification and a **FastAPI backend** to process predictions.

The project combines **mobile development, deep learning, computer vision, and API development** into a single end-to-end application.

---

## 🚀 Overview

Crop diseases can significantly affect agricultural productivity. Early identification of plant diseases can help farmers take appropriate action before the disease spreads.

AgroAI provides a simple workflow:

```text
📷 Capture / Select Leaf Image
            ↓
      📱 Mobile App
            ↓
       FastAPI Backend
            ↓
      🧠 MobileNetV2
            ↓
    Disease Classification
            ↓
       📊 Prediction


Features
- 📷 Plant leaf image selection
- 🧠 Deep learning based disease classification
- ⚡ FastAPI inference backend
- 📱 React Native mobile application
- 🔬 MobileNetV2 architecture
- 🌿 Plant disease recognition
- 🔄 Mobile app → API → ML model workflow
- 📊 Prediction response from the backend


🏗️ Project Architecture

AgroAI
│
├── mobile_app/
│   ├── React Native application
│   └── Mobile UI
│
├── backend/
│   ├── FastAPI API
│   ├── ML model
│   └── Prediction logic
│
├── model/
│   └── MobileNetV2 trained model
│
└── README.md

🧠 Machine Learning
AgroAI uses MobileNetV2, a lightweight convolutional neural network architecture
designed for efficient image classification.
The model is used to classify plant leaf images into disease categories.

Model Pipeline

Input Image
     ↓
Image Preprocessing
     ↓
MobileNetV2
     ↓
Feature Extraction
     ↓
Classification Layer
     ↓
Disease Prediction

MobileNetV2 was selected because its lightweight
architecture makes it suitable for applications where computational efficiency is important.

📱 Mobile Application

The mobile application is built using:
- React Native
- Expo
The application provides the user interface for selecting or
capturing a plant image and communicating with the backend.

Application Flow

User
 │
 ├── Select/Capture Leaf Image
 │
 ↓
React Native App
 │
 ↓
HTTP Request
 │
 ↓
FastAPI Backend
 │
 ↓
MobileNetV2 Model
 │
 ↓
Prediction
 │
 ↓
Result displayed in App
