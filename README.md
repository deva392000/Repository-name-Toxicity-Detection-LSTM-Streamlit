# Deep Learning for Comment Toxicity Detection

## Project Overview

This project uses Deep Learning and an LSTM model to detect toxic comments.

The model performs multi-label classification and predicts six toxicity categories:

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

## Technologies Used

* Python 3.13.15
* TensorFlow 2.20.0
* Keras
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Jupyter Notebook
* Git & GitHub

## Model Architecture

```text
Input Text
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense Layer
    ↓
Sigmoid Output
    ↓
6 Toxicity Predictions
```

## Model Details

* Python Version: 3.13.15
* TensorFlow Version: 2.20.0
* Model: LSTM
* Vocabulary Size: 20,000
* Sequence Length: 200
* Embedding Dimension: 128
* LSTM Units: 64
* Dense Units: 32
* Output Classes: 6
* Loss: Binary Crossentropy
* Optimizer: Adam

## Toxicity Categories

The model predicts the following six categories:

| Category      | Description                            |
| ------------- | -------------------------------------- |
| Toxic         | General toxic language                 |
| Severe Toxic  | Highly toxic language                  |
| Obscene       | Obscene or offensive language          |
| Threat        | Threatening language                   |
| Insult        | Insulting language                     |
| Identity Hate | Hate directed toward an identity group |

## Dataset

The dataset contains comments with multiple toxicity labels.

### Training Dataset

* Rows: 159,571
* Columns: 8

### Testing Dataset

* Rows: 153,164
* Columns: 2

The training dataset contains the comment text and six toxicity labels.

## Evaluation

The model achieved approximately:

* Macro F1 Score: 0.51
* Weighted F1 Score: 0.71

## Streamlit Application

The trained LSTM model is deployed using Streamlit.

Users can enter a comment and receive real-time predictions for each toxicity category.

The application also displays:

* Toxicity probabilities
* Predicted categories
* Highest class probability
* Overall prediction result

## Project Structure

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

## How to Run

Install the required libraries and run the Streamlit application:

```bash
py -3.13 -m streamlit run app.py
```

The application will open in your web browser.

## Prediction Example

Example input:

```text
You are doing a great job!
```

The model analyzes the comment and predicts the probability of each toxicity category.

## Key Learning

Through this project, I learned:

* Text preprocessing
* Tokenization
* Sequence padding
* Deep Learning
* LSTM networks
* Multi-label classification
* Model evaluation
* Threshold tuning
* Model saving and loading
* Streamlit deployment
* Git and GitHub

## Author

Devendra
