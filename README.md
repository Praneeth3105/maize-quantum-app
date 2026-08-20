# 🌽 Quantum Maize Disease Detection System

A cloud-based intelligent web application for maize leaf disease detection using Deep Learning, Quantum Machine Learning, and Firebase.

The application allows users to upload a maize leaf image and predicts whether the leaf is healthy or diseased. If a disease is detected, the system recommends appropriate remedies, dosage, and preventive measures.

---

## 🚀 Features

- Upload maize leaf images
- Automatic image preprocessing
- NASNetMobile feature extraction
- PCA dimensionality reduction
- Quantum Feature Mapping using PennyLane
- Binary Healthy/Disease Classification
- Multi-class Disease Classification
- Disease confidence score
- Firebase Firestore integration
- Automatic prediction history storage
- Disease remedies and prevention suggestions
- Cloud deployment using Streamlit

---

## 🛠 Technologies Used

### Frontend

- Streamlit

### Backend

- Python
- TensorFlow
- OpenCV
- NumPy

### Machine Learning

- NASNetMobile
- TensorFlow/Keras
- PCA
- Scikit-learn
- Joblib

### Quantum Computing

- PennyLane
- Default Qubit Simulator

### Database

- Firebase Firestore

### Deployment

- Streamlit Community Cloud

---

## 📂 Project Structure

```
maize-quantum-app/
│
├── app.py
├── binary_model.h5
├── multi_model.h5
├── scaler_binary.pkl
├── scaler_multi.pkl
├── pca_binary.pkl
├── quantum_params_binary.npy
├── class_names.json
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
├── .github/
│   └── workflows/
│       └── keep_alive.yml
└── README.md
```

---

## ⚙️ Workflow

1. User uploads a maize leaf image.
2. Image is resized to 224×224 pixels.
3. NASNetMobile extracts deep features.
4. Features are normalized.
5. PCA reduces dimensionality.
6. Quantum Feature Mapping generates quantum representations.
7. Binary classifier predicts Healthy or Diseased.
8. Multi-class classifier identifies the disease.
9. Firebase retrieves remedy information.
10. Prediction is stored in Firestore.
11. Results are displayed in Streamlit.

---

## 🧠 Machine Learning Pipeline

```
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
NASNetMobile Feature Extraction
      │
      ▼
Feature Scaling
      │
      ▼
PCA
      │
      ▼
Quantum Feature Mapping
      │
      ▼
Binary Classification
      │
      ▼
Healthy / Diseased
      │
      ▼
Multi-class Classification
      │
      ▼
Disease Prediction
      │
      ▼
Firebase
      │
      ▼
Treatment Recommendation
```

---

## 📋 Disease Detection

The system detects:

- Healthy Leaf
- Leaf Blight
- Leaf Spot
- Common Rust
- Gray Leaf Spot
- Northern Leaf Blight

*(Update this list based on your trained dataset.)*

---

## 💊 Treatment Recommendation

For every detected disease, the application provides:

- Recommended Remedy
- Dosage
- Prevention Measures

The information is retrieved directly from Firebase Firestore.

---

## ☁️ Firebase Integration

Firestore stores:

### Predictions

- Disease Name
- Confidence Score
- Timestamp

### Remedies

- Disease Name
- Remedy
- Dosage
- Prevention

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/Praneeth3105/maize-quantum-app.git
```

Move into the project

```bash
cd maize-quantum-app
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🔐 Environment Variables

Create:

```
.streamlit/secrets.toml
```

Add your Firebase credentials:

```toml
[firebase]

type=""
project_id=""
private_key_id=""
private_key=""
client_email=""
client_id=""
...
```

---

## 🌐 Live Application

You can access the deployed application here:

**🔗 Live Demo:**  
https://maize-quantum-app-6jmirwun4wa2aokjqnshgh.streamlit.app/

> **Note:** This application is hosted on **Streamlit Community Cloud**. If the app has been inactive for some time, it may enter sleep mode to conserve resources.
>
> If you see that the application is unavailable, simply click the **"Wake up"** button displayed on the page. Streamlit will automatically restart the application, which usually takes **30–60 seconds**. Once the app is awake, refresh the page if necessary and continue using it normally.

---
