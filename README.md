
# 🔥 Fire Detection from Images using Convolutional Neural Networks

## 📋 Project Overview

Early and accurate detection of fire is crucial for preventing disasters and ensuring public safety. Traditional fire detection methods (e.g., smoke detectors or sensor-based systems) may be limited by environmental conditions, false alarms, or slow detection rates.  
This project leverages **Deep Learning** techniques to automatically detect fire from images using a **Convolutional Neural Network (CNN)** classifier.

## 🎯 Objective

The objective of this project is to build and train a CNN model capable of classifying images into two categories:
- **Fire**
- **No Fire**

This AI model can potentially assist in real-time surveillance systems by automatically flagging hazardous fire events.

## 📊 Dataset

The dataset used in this project is the **Fire Dataset** from [Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset).  
It contains labeled images distributed into two classes:
- **Fire**: Images containing fire.
- **No Fire**: Images without fire.

The dataset is organized into:
- `train/`
- `validation/`
- `test/`

## 🛠️ Project Structure

```
├── FireDetection.ipynb     # Main Jupyter notebook
├── README.md               # Project overview (this file)
├── my_model.keras          # Saved trained model (best version)
├── Dataset/
│   ├── train/
│   └── test/
```

## ⚙️ Methodology

- **Data Augmentation** to improve generalization and prevent overfitting
- **CNN Architecture** with:
  - Convolutional layers + MaxPooling
- **Callbacks**:
  - `ModelCheckpoint` to save the best model
  - `ReduceLROnPlateau` to adjust learning rate dynamically

## 🧩 Model Architecture

The CNN model includes:
- Stacked `Conv2D` layers with ReLU activation
- `MaxPooling2D` layers
- `BatchNormalization` and `Dropout` applied throughout
- Fully connected `Dense` layers before output
- `Sigmoid` output activation for binary classification

## 📈 Evaluation

Model performance is evaluated using:
- Test accuracy and loss
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)

## 🚀 How to Run

1. Clone the repository and install dependencies

```bash
pip install tensorflow scikit-learn matplotlib
```

2. Run the notebook `FireDetection.ipynb` to:
- Preprocess data
- Train the model
- Evaluate the model
- Visualize metrics

3. To load and evaluate the best saved model:

```python
from tensorflow.keras.models import load_model
model = load_model('my_model.keras')
```

## ✅ Results

The model achieves high accuracy on unseen test data and demonstrates strong generalization for binary fire detection tasks.  
Evaluation metrics, including the confusion matrix and classification report, confirm the model's reliability.

## 📚 References

- [Fire Dataset - Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- TensorFlow Documentation
- Scikit-learn Documentation

## ✨ Future Work

- Deploy the model into a real-time surveillance system (e.g., CCTV monitoring)
- Extend to video fire detection
- Experiment with transfer learning using pre-trained models

## 👨‍💻 Author

Mohamed Amine Lachaal
