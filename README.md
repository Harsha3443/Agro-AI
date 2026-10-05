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


⚡ Backend

The backend is implemented using FastAPI.
Its main responsibility is to:
1. Receive the plant image.
2. Preprocess the image.
3. Pass the image to the trained model.
4. Perform inference.
5. Return the prediction to the mobile application.

Example API workflow:

POST /predict
        ↓
Image Upload
        ↓
Preprocessing
        ↓
Model Inference
        ↓
Prediction Response


🛠️ Tech Stack

Mobile
- React Native
- Expo
Backend
- Python
- FastAPI
Machine Learning
- TensorFlow
- MobileNetV2
- Convolutional Neural Networks
Development Tools
- VS Code
- Git
- GitHub
- Expo / EAS

📂 Dataset

The project uses the PlantVillage dataset for plant disease classification.
The dataset contains images of plant leaves belonging to different plant and disease categories.

🔬 Key Concepts Used

This project demonstrates practical implementation of:
- Computer Vision
- Image Classification
- Convolutional Neural Networks
- Transfer Learning
- MobileNetV2
- Model Inference
- REST APIs
- FastAPI
- React Native
- Mobile-to-Backend Communication

▶️ Running the Project

1. Clone the repository
git clone https://github.com/Harsha3443/Crop-Disease-Detection.git

cd Crop-Disease-Detection

2. Backend Setup
Navigate to the backend directory:
cd backend

Create a virtual environment:
python -m venv venv

Activate it on Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Start the FastAPI server:
uvicorn main:app --reload

The API will then be available locally.
3. Mobile App Setup
Open another terminal and navigate to the mobile application:
cd mobile_app

Install dependencies:
npm install

Start Expo:
npx expo start

Then run the application using an available Expo development environment.

🔐 Configuration

If the mobile application communicates with a locally running backend, update the API base URL according to your development environment.
For example:
http://YOUR_LOCAL_IP:8000

Make sure the mobile device and backend machine can communicate over the same network when testing locally.

🎯 Future Improvements

Potential improvements include:
- 🌿 Supporting additional crops and diseases
- 📈 Improving model accuracy
- ⚡ Optimizing inference speed
- 📱 On-device inference
- 🌐 Cloud deployment
- 📊 Confidence visualization
- 🧑‍🌾 Providing disease-specific recommendations
- 🌍 Multilingual support
- 📡 Offline prediction capability

👨‍💻 Author

Harsha Vardhan Sai
B.Tech — Artificial Intelligence & Machine Learning
GitHub:
https://github.com/Harsha3443
