# 🧠 Deep Learning for Comment Toxicity Detection

A **Deep Learning-based multi-label text classification project** that detects different types of toxic comments using an **LSTM neural network** and provides real-time predictions through a **Streamlit web application**.

## 📌 Project Overview

Online platforms receive a large number of user comments, making manual moderation difficult.

This project uses an **LSTM (Long Short-Term Memory)** model to automatically classify comments into six different toxicity categories.

The trained model is integrated with Streamlit to provide an interactive interface where users can enter a comment and instantly view the prediction results.

## 🎯 Toxicity Categories

The model predicts six categories:

* **Toxic**
* **Severe Toxic**
* **Obscene**
* **Threat**
* **Insult**
* **Identity Hate**

Since a single comment can belong to multiple categories, this project uses **Multi-Label Classification**.

---

## 🛠️ Technologies Used

* **Python 3.12.13** — Model development & Jupyter Notebook
* **Python 3.13.15** — Streamlit application environment
* **TensorFlow 2.20.0**
* **Keras**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

---

## 🧠 Deep Learning Workflow

```text
Raw Comments
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Sequence Padding
     ↓
Train / Validation Split
     ↓
Embedding Layer
     ↓
LSTM Layer
     ↓
Dense Layer
     ↓
Sigmoid Output Layer
     ↓
6 Toxicity Predictions
```

---

## 🏗️ Model Architecture

```text
Input Text
    ↓
Embedding Layer
    ↓
LSTM (64 Units)
    ↓
Dense Layer (32 Units)
    ↓
Sigmoid Output (6 Classes)
```

### Model Configuration

| Parameter                | Value               |
| ------------------------ | ------------------- |
| Model Development Python | 3.12.13             |
| Streamlit Python         | 3.13.15             |
| TensorFlow               | 2.20.0              |
| Model                    | LSTM                |
| Vocabulary Size          | 20,000              |
| Sequence Length          | 200                 |
| Embedding Dimension      | 128                 |
| LSTM Units               | 64                  |
| Dense Units              | 32                  |
| Output Classes           | 6                   |
| Loss Function            | Binary Crossentropy |
| Optimizer                | Adam                |

---

## 📊 Dataset

The project uses a labeled comment toxicity dataset.

### Training Data

* **Rows:** 159,571
* **Columns:** 8

### Testing Data

* **Rows:** 153,164
* **Columns:** 2

The training data contains the comment text and six toxicity labels.

### Target Labels

```text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

---

## 📈 Model Evaluation

The trained model achieved approximately:

| Metric            |    Score |
| ----------------- | -------: |
| Macro F1 Score    | **0.51** |
| Weighted F1 Score | **0.71** |

Class-specific probability thresholds were also tuned to improve multi-label predictions.

---

## 🌐 Streamlit Application

The trained LSTM model is deployed using **Streamlit**.

The application allows users to:

* Enter a comment
* Analyze the comment in real time
* View toxicity probabilities
* Identify predicted toxicity categories
* View the highest class probability
* Compare probabilities across all six categories

---

## 📁 Project Structure

```text
Toxicity Detection/
│
├── app.py
├── toxicity.ipynb
├── toxicity_lstm_model.keras
├── tokenizer.pkl
├── train.csv
├── test.csv
├── test_predictions.csv
├── README.md
└── .gitignore
```

---

## ▶️ How to Run the Application

### 1. Clone the Repository

```bash
git clone https://github.com/deva392000/Repository-name-Toxicity-Detection-LSTM-Streamlit.git
```

### 2. Open the Project Folder

```bash
cd Repository-name-Toxicity-Detection-LSTM-Streamlit
```

### 3. Run the Streamlit Application

```bash
py -3.13 -m streamlit run app.py
```

The application will open in your browser.

---

## 📚 Key Learning Outcomes

Through this project, I learned:

* Text preprocessing
* Tokenization
* Sequence padding
* Word embeddings
* LSTM neural networks
* Multi-label classification
* Model training and validation
* F1-score evaluation
* Probability threshold tuning
* Model saving and loading
* Streamlit deployment
* Git and GitHub

---

## 🚀 Future Improvements

* Experiment with **Bidirectional LSTM**
* Experiment with **GRU and Transformer models**
* Improve handling of class imbalance
* Optimize classification thresholds
* Improve model performance
* Deploy the application on a cloud platform

---

## 👨‍💻 Author

**Devendra**

Built as a Deep Learning project demonstrating **NLP, LSTM, Multi-Label Classification, Model Evaluation, and Streamlit Deployment**.
