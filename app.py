
import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Toxicity Detection",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND TOKENIZER
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("toxicity_lstm_model.keras")


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()
tokenizer = load_tokenizer()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# LABELS AND THRESHOLDS
# =========================================================

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

thresholds = [0.5, 0.5, 0.5, 0.1, 0.5, 0.1]


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_toxicity(text):

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    predictions = probabilities >= np.array(thresholds)

    return probabilities, predictions


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🧠 Comment Toxicity Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based Multi-Label Toxicity Classification using LSTM'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INFORMATION CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🧠 Model", "LSTM")

with col2:
    st.metric("🏷️ Labels", "6")

with col3:
    st.metric("🎯 Task", "Multi-Label")

with col4:
    st.metric("⚡ Prediction", "Real-Time")


# =========================================================
# COMMENT INPUT
# =========================================================

st.markdown("---")

left_col, right_col = st.columns([1.5, 1])


with left_col:

    st.subheader("💬 Enter Your Comment")

    comment = st.text_area(
        "Type a comment below:",
        height=180,
        placeholder="Enter a comment to analyze..."
    )

    st.write("### 🧪 Sample Comments")

    sample_col1, sample_col2, sample_col3 = st.columns(3)

    with sample_col1:
        positive = st.button("😊 Positive")

    with sample_col2:
        toxic = st.button("😡 Toxic")

    with sample_col3:
        neutral = st.button("😐 Neutral")

    if positive:
        comment = "Thank you for your amazing work!"

    if toxic:
        comment = "You are stupid and I hate you!"

    if neutral:
        comment = "The weather is nice today."


    analyze = st.button(
        "🔍 Analyze Comment",
        use_container_width=True
    )


with right_col:

    st.subheader("ℹ️ About the Model")

    st.write("""
    This application uses an LSTM-based Deep Learning model
    to classify comments into six different toxicity categories.

    The model performs multi-label classification, meaning one
    comment can belong to multiple toxicity categories.
    """)

    st.write("### Model Details")

    st.write("""
    - Model: LSTM
    - Vocabulary Size: 20,000
    - Sequence Length: 200
    - Embedding Dimension: 128
    - LSTM Units: 64
    - Dense Units: 32
    - Output Classes: 6
    - Loss: Binary Crossentropy
    - Optimizer: Adam
    """)


# =========================================================
# PREDICTION RESULTS
# =========================================================

if analyze:

    if comment.strip() == "":
        st.warning("⚠️ Please enter a comment.")

    else:

        probabilities, predictions = predict_toxicity(comment)

        st.markdown("---")

        st.header("📊 Prediction Results")

        # Highest probability
        max_probability = np.max(probabilities)

        st.write("### Highest Class Probability")

        st.write(
            f"Highest class probability: "
            f"**{max_probability * 100:.2f}%**"
        )


        # Overall result
        if np.any(predictions):

            st.error(
                "🚨 Toxic content detected"
            )

        else:

            st.success(
                "✅ No toxic content detected"
            )


        # Category results
        st.write("### 🏷️ Toxicity Categories")

        for label, probability, prediction in zip(
            labels,
            probabilities,
            predictions
        ):

            percentage = probability * 100

            if prediction:

                st.error(
                    f"🚨 {label.replace('_', ' ').title()} "
                    f"— {percentage:.2f}%"
                )

            else:

                st.write(
                    f"✅ {label.replace('_', ' ').title()} "
                    f"— {percentage:.2f}%"
                )

            st.progress(
                float(probability)
            )


# =========================================================
# DATA INSIGHTS
# =========================================================

st.markdown("---")

st.header("📊 Data Insights")

st.write("### Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Comments",
        "159,571"
    )

with col2:
    st.metric(
        "Toxic Comments",
        "15,294"
    )

with col3:
    st.metric(
        "Toxicity Labels",
        "6"
    )


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

st.write("### 📈 Toxicity Label Distribution")

label_counts = {
    "Toxic": 15294,
    "Severe Toxic": 1595,
    "Obscene": 8449,
    "Threat": 478,
    "Insult": 7877,
    "Identity Hate": 1405
}

st.bar_chart(label_counts)


# =========================================================
# LABEL PERCENTAGES
# =========================================================

st.write("### 📊 Label Percentages")

label_percentages = {
    "Toxic": 9.58,
    "Severe Toxic": 1.00,
    "Obscene": 5.29,
    "Threat": 0.30,
    "Insult": 4.94,
    "Identity Hate": 0.88
}

for label, percentage in label_percentages.items():

    st.write(
        f"**{label}: {percentage}%**"
    )

    st.progress(
        percentage / 100
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.header("📈 Model Performance")

metric_col1, metric_col2 = st.columns(2)

with metric_col1:

    st.metric(
        "Macro F1 Score",
        "0.51"
    )

with metric_col2:

    st.metric(
        "Weighted F1 Score",
        "0.71"
    )
# =========================================================
# BULK CSV PREDICTION
# =========================================================

st.markdown("---")

st.header("📁 Bulk CSV Prediction")

st.write(
    "Upload a CSV file containing a "
    "`comment_text` column."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    import pandas as pd

    df = pd.read_csv(uploaded_file)

    st.write("### 📋 Uploaded Data")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # Check comment_text column
    if "comment_text" not in df.columns:

        st.error(
            "❌ CSV must contain a 'comment_text' column."
        )

    else:

        st.success(
            f"✅ {len(df)} comments found."
        )

        if st.button(
            "🔍 Predict All Comments",
            use_container_width=True
        ):

            results = []

            progress = st.progress(0)

            total = len(df)

            for i, text in enumerate(df["comment_text"]):

                probabilities, predictions = predict_toxicity(
                    str(text)
                )

                result = {
                    "comment_text": text
                }

                for label, probability, prediction in zip(
                    labels,
                    probabilities,
                    predictions
                ):

                    result[f"{label}_probability"] = round(
                        float(probability),
                        4
                    )

                    result[label] = int(prediction)

                results.append(result)

                progress.progress(
                    (i + 1) / total
                )

            result_df = pd.DataFrame(results)

            st.success(
                "✅ Bulk prediction completed!"
            )

            st.write("### 📊 Prediction Results")

            st.dataframe(
                result_df,
                use_container_width=True
            )

            csv_data = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="toxicity_bulk_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        🧠 Deep Learning • LSTM • NLP • Multi-Label Classification
        <br>
        Built with Python, TensorFlow & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)