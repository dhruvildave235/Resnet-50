# Federated Learning for EuroSAT Land Use Classification

This project implements a **Federated Learning (FL)** framework using **Flower (flwr)** to classify satellite images from the **EuroSAT dataset** into **10 land-use/land-cover classes**.  
Each client trains a local deep learning model, and a central server aggregates the learned weights using **FedAvg** without sharing raw data.

---

## 📊 Dataset

- **Dataset**: EuroSAT
- **Number of Classes**: 10  
- **Image Size**: 224 × 224 (RGB)
- **Data Split**:
  - Train: 60%
  - Validation: 20%
  - Test: 20%

Each client uses its own local dataset:
- `dataset1` → Client 1
- `dataset2` → Client 2

---

## 🧠 Model Architecture

- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Dense (256, ReLU)
  - Dropout (0.5)
  - Dense (10, Softmax)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

---

## ⚙️ Federated Learning Setup

- **Framework**: Flower (flwr)
- **Strategy**: Federated Averaging (FedAvg)
- **Clients**: 2
- **Server Address**: `localhost:8080`
- **Early Stopping** enabled for efficient training

---

## 🚀 How to Run the Project

### 1️⃣ Start the Federated Server
```bash
python server.py
```
### 2️⃣ Start Client 1 (in a new terminal)
```bash
python client1.py
```
### 3️⃣ Start Client 2 (in a new terminal)
```bash
python client2.py
```
## 📈 Outputs
### Each client saves:
 >   Trained model (final_model.h5)
 >   Training metrics (training_metrics.csv)
 >   Accuracy and loss plots
 >   Final evaluation metrics:
     >   Accuracy
     >   Precision
     >   Recall
     >   F1-score
     >   Confusion Matrix


## 📌 Use Case

Remote sensing image classification
Privacy-aware satellite image analysis
Distributed AI training environments

## ✍️ Created By Dhruvil_Dave

AI & Machine Learning Enthusiast | Software Developer | Research-Oriented Innovator


