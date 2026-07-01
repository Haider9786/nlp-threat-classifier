"""Streamlit UI for the phishing email classifier."""
import joblib
import streamlit as st

from threat_classifier.preprocess import clean_text, ensure_nltk_data
from threat_classifier.utils import load_config, get_logger

st.set_page_config(page_title="Threat Intel Classifier", page_icon="🛡️")

config = load_config()
logger = get_logger(__name__, config)
model_cfg = config["model"]


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer, with a clear error if missing."""
    try:
        ensure_nltk_data()
        model = joblib.load(model_cfg["best_model_path"])
        vectorizer = joblib.load(model_cfg["vectorizer_path"])
        return model, vectorizer
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        st.error(
            "Model files not found. Run train.py first to generate "
            "models/best_model.pkl and models/vectorizer.pkl."
        )
        st.stop()


model, vectorizer = load_model()

st.title("🛡️ NLP Threat Intelligence Classifier")
st.markdown(
    "Paste email or threat intel text below to classify it as **Safe** or **Phishing**"
)

user_input = st.text_area(
    "Email / Threat Intel Text",
    height=200,
    placeholder="Paste suspicious email content here...",
)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        if cleaned == "":
            st.warning("No usable text after cleaning — try a longer message.")
        else:
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            if prediction == 1:
                st.error("⚠️ PHISHING DETECTED")
            else:
                st.success("✅ SAFE")

            with st.expander("See cleaned text used for prediction"):
                st.code(cleaned)

st.markdown("---")
st.markdown(
    "**Model:** Logistic Regression (tuned, C=10) with TF-IDF | "
    "**Test Accuracy:** 96.75% | 5-fold CV | Trained on 18,634 real phishing/safe emails"
)