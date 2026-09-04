# ============================================================
# COMMENT TOXICITY DETECTION
# Deep Learning - LSTM + Streamlit
# ============================================================

import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid #ddd;
}

.safe {
    background-color: #f0fff4;
}

.danger {
    background-color: #fff5f5;
}

.score {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #888;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL AND TOKENIZER
# ============================================================

@st.cache_resource
def load_model_and_tokenizer():

    model = tf.keras.models.load_model(
        "toxicity_lstm_model.keras"
    )

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Keep alphabets and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 5. LABELS
# ============================================================

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ============================================================
# 6. THRESHOLDS
# ============================================================

thresholds = [
    0.5,
    0.5,
    0.5,
    0.1,
    0.5,
    0.1
]


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_toxicity(comment):

    # Clean comment
    comment = clean_text(comment)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [comment]
    )

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    # Model prediction
    probabilities = model.predict(
        padded,
        verbose=0
    )[0]

    return probabilities


# ============================================================
# 8. HEADER
# ============================================================

st.markdown(
    '<div class="title">🛡️ Comment Toxicity Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered toxic comment detection using Deep Learning LSTM'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 9. INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🤖 Model",
        "LSTM"
    )

with col2:
    st.metric(
        "🏷️ Labels",
        "6"
    )

with col3:
    st.metric(
        "🧠 Task",
        "Multi-Label"
    )

with col4:
    st.metric(
        "⚡ Prediction",
        "Real-Time"
    )


st.markdown("---")


# ============================================================
# 10. MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [1.4, 1]
)


# ============================================================
# 11. COMMENT INPUT
# ============================================================

with left:

    st.subheader("💬 Enter Your Comment")

    comment = st.text_area(
        "Comment",
        height=200,
        placeholder="Example: You are a wonderful person...",
        label_visibility="collapsed"
    )


    # Example comments

    st.write("**Try an example:**")

    example1, example2, example3 = st.columns(3)

    with example1:

        if st.button(
            "😊 Positive",
            use_container_width=True
        ):
            comment = "You are a wonderful person"

    with example2:

        if st.button(
            "😡 Toxic",
            use_container_width=True
        ):
            comment = "You are stupid and disgusting"

    with example3:

        if st.button(
            "😐 Neutral",
            use_container_width=True
        ):
            comment = "The weather is nice today"


    st.write("")


    # Predict button

    predict_button = st.button(
        "🔍 Analyze Comment",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# 12. ABOUT MODEL
# ============================================================

with right:

    st.subheader("ℹ️ About This Model")

    st.write(
        "This application uses an LSTM-based "
        "Deep Learning model to identify different "
        "types of toxic comments."
    )

    st.write("**Detected Categories:**")

    st.write("🔴 Toxic")
    st.write("🔴 Severe Toxic")
    st.write("🔴 Obscene")
    st.write("🔴 Threat")
    st.write("🔴 Insult")
    st.write("🔴 Identity Hate")

    st.info(
        "One comment can belong to multiple categories. "
        "Therefore, this is a multi-label classification problem."
    )


# ============================================================
# 13. PREDICTION
# ============================================================

if predict_button:

    if comment.strip() == "":

        st.warning(
            "⚠️ Please enter a comment first."
        )

    else:

        # Get probabilities

        probabilities = predict_toxicity(
            comment
        )


        # ====================================================
        # OVERALL SCORE
        # ====================================================

        max_probability = float(
            np.max(probabilities)
        )

        toxic_detected = any(
            probability >= threshold
            for probability, threshold
            in zip(
                probabilities,
                thresholds
            )
        )


        st.markdown("---")

        st.subheader("📊 Analysis Result")


        # ====================================================
        # OVERALL RESULT
        # ====================================================

        if toxic_detected:

            st.error(
                "🚨 Toxic Content Detected"
            )

        else:

            st.success(
                "✅ No Toxic Content Detected"
            )


        # ====================================================
        # TOXICITY SCORE
        # ====================================================

        st.write("### Highest Class Probability")

        st.progress(
            min(max_probability, 1.0)
        )

        st.write(
    f"Highest class probability: "
    f"**{max_probability * 100:.2f}%**"
)


        # ====================================================
        # CATEGORY RESULTS
        # ====================================================

        st.write("### 🏷️ Category Predictions")


        result_columns = st.columns(2)


        for i, (
            label,
            probability,
            threshold
        ) in enumerate(
            zip(
                labels,
                probabilities,
                thresholds
            )
        ):

            prediction = (
                probability >= threshold
            )

            percentage = (
                probability * 100
            )


            with result_columns[i % 2]:

                if prediction:

                    st.error(
                        f"🚨 {label.replace('_', ' ').title()}"
                    )

                else:

                    st.success(
                        f"✅ {label.replace('_', ' ').title()}"
                    )

                st.progress(
                    float(probability)
                )

                st.caption(
                    f"Probability: "
                    f"{percentage:.2f}%"
                )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Deep Learning Project • LSTM Multi-Label Classification • '
    'Streamlit'
    '</div>',
    unsafe_allow_html=True
)