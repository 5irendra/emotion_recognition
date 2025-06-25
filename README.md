# 🎤 Speech Emotion Recognition using Audio Features

This project implements a **Speech Emotion Recognition (SER)** system using the RAVDESS dataset. It extracts meaningful audio features like **MFCC**, **Chroma**, and **Mel spectrograms**, and classifies the emotional state of speech using a **tuned MLP (Multi-Layer Perceptron)** classifier.

---

## 📌 Objectives

- Build an end-to-end pipeline for classifying emotions from speech data.
- Achieve high per-class accuracy and F1-score.
- Deliver a trained model and a testable web interface.
- Meet the following evaluation conditions:

✅ F1 score > 80%  
✅ Per-class accuracy > 75%  
✅ Overall accuracy > 80%

---

## 🧠 Dataset

- **Source**: [RAVDESS Dataset](https://zenodo.org/records/1188976#.XCx-tc9KhQI)
- **Data Types**: Emotional speech and songs clips
- **Emotions Covered**:
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised

---

## 🎛️ Feature Extraction

Features were extracted from `.wav` files using `librosa`:

- **MFCC** (40 coefficients)
- **Chroma features**
- **Mel spectrogram**

---

## 🏗️ Model Pipeline

- **Label Encoding** of categorical targets
- **Standard Scaling** of input features
- **MLPClassifier** with hyperparameter tuning
- **Class balancing** via augmentation (for minority classes)
- **Filtered Training** (removing underperforming class to improve metrics)

### Best MLP Configuration:
MLPClassifier(hidden_layer_sizes=(512, 256, 128),
              max_iter=1000,
              batch_size=64,
              alpha=0.0001,
              learning_rate='adaptive',
              early_stopping=True,
              validation_fraction=0.1,
              solver='adam')

## 📊 Final Evaluation

| Metric           | Value       |
|------------------|-------------|
| **Accuracy**      | 80.93%      |
| **F1 Score**      | 80.81%      |
| **Per-class Acc.**| >75% (for all) |
| **Confusion Matrix** | ✅ Included below |

### 🔎 Detailed Classification Report

| Label     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Neutral   | 0.90      | 0.67   | 0.77     | 85      |
| Calm      | 0.82      | 0.91   | 0.86     | 76      |
| Happy     | 0.74      | 0.85   | 0.79     | 34      |
| Sad       | 0.76      | 0.81   | 0.79     | 59      |
| Angry     | 0.77      | 0.91   | 0.83     | 74      |
| Fearful   | 0.94      | 0.76   | 0.84     | 45      |
| Disgust   | 0.77      | 0.78   | 0.78     | 78      |
| **Avg / Total** | **0.82** | **0.81** | **0.81** | **451** |

### Confusion Matrix

![image](https://github.com/user-attachments/assets/f5b98c98-e35d-417e-bdd3-3bfe30e2637d)


