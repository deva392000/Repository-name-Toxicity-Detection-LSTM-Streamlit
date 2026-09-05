# 🧠 Deep Learning for Comment Toxicity Detection

A **Deep Learning-based multi-label text classification project** that detects different types of toxic comments using an **LSTM (Long Short-Term Memory) neural network** and provides real-time predictions through a **Streamlit web application**.

---

## 📌 Project Overview

Online platforms receive a large number of user comments, making manual moderation difficult.

This project uses an **LSTM-based Deep Learning model** to automatically classify comments into six different toxicity categories.

The trained model is integrated with Streamlit to provide an interactive interface where users can:

* Enter a comment
* Get real-time toxicity predictions
* View probability scores
* Identify detected toxicity categories
* Upload a CSV file for bulk prediction
* Download prediction results as a CSV file

---

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

* **Python 3.12.13** — Model development and Jupyter Notebook
* **Python 3.13.15** — Streamlit application environment
* **TensorFlow 2.20.0**
* **Keras**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
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
Stopword Removal
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

## 🧹 Text Preprocessing

The following preprocessing steps are performed:

1. Convert text to lowercase
2. Remove URLs
3. Remove HTML tags
4. Remove special characters and numbers
5. Remove extra spaces
6. Remove English stopwords
7. Tokenize text into numerical sequences
8. Pad sequences to a fixed length of 200

Example:

```text
Original:
D'aww! He matches this background colour I'm seemingly stuck with. Thanks.

After cleaning:
daww he matches this background colour im seemingly stuck with thanks

After stopword removal:
daww matches background colour im seemingly stuck thanks
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
| Epochs                   | 5                   |
| Batch Size               | 64                  |

---

## 📊 Dataset

The project uses a labeled comment toxicity dataset.

### Training Data

* **Rows:** 159,571
* **Columns:** 8

### Testing Data

* **Rows:** 153,164
* **Columns:** 2

### Training Columns

```text
id
comment_text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

### Testing Columns

```text
id
comment_text
```

---

## 🎯 Target Labels

```text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

The six labels are binary values (`0` or `1`).

Because multiple labels can be `1` for the same comment, this is a **multi-label classification problem**.

---

## 📈 Model Evaluation

The final LSTM model achieved the following results on the validation dataset:

| Metric              |      Score |
| ------------------- | ---------: |
| Training Accuracy   | **99.34%** |
| Validation Accuracy | **99.40%** |
| Macro F1 Score      |   **0.44** |
| Weighted F1 Score   |   **0.69** |

### Class-wise F1 Scores

| Category      | F1 Score |
| ------------- | -------: |
| Toxic         | **0.77** |
| Severe Toxic  | **0.15** |
| Obscene       | **0.79** |
| Threat        | **0.02** |
| Insult        | **0.67** |
| Identity Hate | **0.24** |

The lower performance for some categories is mainly related to **class imbalance**, because categories such as `threat`, `severe_toxic`, and `identity_hate` contain far fewer positive examples.

---

## 🎚️ Classification Thresholds

Different probability thresholds were used for the six categories:

```text
Toxic          → 0.5
Severe Toxic   → 0.5
Obscene        → 0.5
Threat         → 0.1
Insult         → 0.5
Identity Hate  → 0.1
```

Lower thresholds were used for some rare categories to improve their ability to detect positive cases.

---

## 🌐 Streamlit Application

The trained LSTM model is integrated into a **Streamlit web application**.

### Features

#### 💬 Single Comment Prediction

Users can:

* Enter a comment
* Analyze it in real time
* View toxicity probabilities
* See detected toxicity categories
* View the highest class probability

#### 🧪 Sample Comments

The application provides sample buttons for:

* 😊 Positive comment
* 😡 Toxic comment
* 😐 Neutral comment

#### 📊 Data Insights

The application displays:

* Total comments
* Toxic comment count
* Six toxicity categories
* Label distribution
* Label percentages

#### 📈 Model Performance

The application displays:

* Training Accuracy
* Validation Accuracy
* Macro F1 Score
* Weighted F1 Score

#### 📁 Bulk CSV Prediction

Users can upload a CSV containing a:

```text
comment_text
```

column.

The application:

1. Reads the CSV
2. Processes each comment
3. Predicts all six toxicity categories
4. Displays prediction results
5. Provides a downloadable CSV file

---

## 📁 Project Structure

```text
Toxicity Detection/
│
├── app.py
├── toxicity.ipynb
│
├── toxicity_lstm_model.keras
├── tokenizer.pkl
│
├── train.csv
├── test.csv
├── test_predictions.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Important Files

| File                        | Purpose                                     |
| --------------------------- | ------------------------------------------- |
| `app.py`                    | Streamlit application                       |
| `toxicity.ipynb`            | Data preprocessing, training and evaluation |
| `toxicity_lstm_model.keras` | Trained LSTM model                          |
| `tokenizer.pkl`             | Saved text tokenizer                        |
| `test_predictions.csv`      | Predictions for test dataset                |
| `requirements.txt`          | Required Python packages                    |
| `README.md`                 | Project documentation                       |
| `.gitignore`                | Files excluded from Git                     |

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

### 3. Install Dependencies

```bash
py -3.13 -m pip install -r requirements.txt
```

### 4. Download NLTK Stopwords

```bash
py -3.13 -c "import nltk; nltk.download('stopwords')"
```

### 5. Run the Streamlit Application

```bash
py -3.13 -m streamlit run app.py
```

The application will open in your browser.

---

## 📚 Key Learning Outcomes

Through this project, I learned:

* Text preprocessing
* Text cleaning
* Stopword removal
* Tokenization
* Sequence padding
* Word embeddings
* LSTM neural networks
* Multi-label classification
* Model training and validation
* Classification report
* F1-score evaluation
* Class-specific probability thresholds
* Model saving and loading
* Test dataset prediction
* CSV generation
* Streamlit application development
* Bulk CSV prediction
* Git and GitHub

---

## 🚀 Future Improvements

Possible improvements include:

* Experiment with **Bidirectional LSTM**
* Experiment with **GRU**
* Experiment with **Transformer models**
* Improve handling of class imbalance
* Optimize classification thresholds using validation data
* Use class weights or focal loss
* Improve rare-class detection
* Optimize bulk prediction performance
* Deploy the application on a cloud platform

---

## 👨‍💻 Author

**Devendra**

Built as a Deep Learning project demonstrating:

**NLP • Text Preprocessing • LSTM • Multi-Label Classification • Model Evaluation • Streamlit Deployment**

---
