# 🔥 Thermal Anomaly Detection using Deep Autoencoders

A modular, unsupervised system for detecting thermal anomalies in infrared images using a convolutional autoencoder combined with classical thermal hotspot analysis.

This project is designed with clean architecture, clear separation of concerns, and production-style structure, making it suitable for academic reviews, hackathons, and industry evaluation.

---

## 📌 Problem Statement

Thermal imaging systems are widely used in applications such as:
- Electrical equipment monitoring
- Industrial fault detection
- Fire and heat leak detection

However, labeled anomaly data is often scarce. This project addresses that limitation using unsupervised learning, where the model learns normal thermal patterns and flags deviations as anomalies.

---

## 🧠 Solution Overview

The system follows a two-stage anomaly detection approach:

### 1️⃣ Deep Learning (Global Anomaly Detection)
- A Convolutional Autoencoder is trained only on normal thermal images.
- Images with high reconstruction error are flagged as anomalous.
- A dynamic percentile-based threshold is used instead of a hardcoded value.

### 2️⃣ Classical Computer Vision (Local Hotspot Analysis)
- Threshold-based hotspot extraction using OpenCV.
- Computes interpretable statistics:
  - Maximum temperature
  - Mean temperature
  - Standard deviation
  - Hotspot area (pixel count)

This hybrid approach improves robustness, interpretability, and reliability.

---

## 📁 Project Structure

thermal_anomaly/
│
├── data.py        # Image loading, preprocessing, thermal statistics
├── model.py       # Convolutional autoencoder definition
├── train.py       # Training loop & anomaly scoring
├── viz.py         # Visualization utilities
├── main.py        # End-to-end pipeline orchestration
└── requirements.txt

---

## ⚙️ Pipeline Flow

Thermal Images
      ↓
Preprocessing (crop, grayscale, normalize)
      ↓
Autoencoder Training (normal data only)
      ↓
Reconstruction Error Scoring
      ↓
Dynamic Thresholding (percentile-based)
      ↓
Anomaly Classification
      ↓
Hotspot Detection & Visualization

---

## 📊 Outputs

- results.csv containing:
  - Image path
  - Anomaly score
  - Prediction (Normal / Anomaly)
  - Thermal statistics (max, mean, std, hotspot area)

- Visual outputs:
  - Original image
  - Reconstruction
  - Pixel-wise anomaly heatmap
  - Hotspot overlay on original image

---

## 🚀 How to Run

### Install Dependencies
```bash
 pip install -r requirements.txt
```
### Run the Pipeline
```bash
python main.py
```
---

## 🛠 Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy / Pandas
- Matplotlib

---

## 📈 Why This Design Works

- No labeled anomalies required
- Scales well to real-world thermal datasets
- Combines deep learning with explainable computer vision
- Clean, maintainable, testable codebase

---

## 🔮 Future Improvements

- Model checkpointing & early stopping
- Temporal anomaly detection for video streams
- ONNX / TensorRT deployment
- REST API for real-time inference
- IoT thermal sensor integration

---

## 📄 License

Intended for educational, research, and prototype use.
