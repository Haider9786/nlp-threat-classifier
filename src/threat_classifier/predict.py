"""Command-line interface for classifying email text without the Streamlit UI."""
import argparse
import sys

import joblib

from threat_classifier.preprocess import clean_text, ensure_nltk_data
from threat_classifier.utils import load_config, get_logger


def predict(text: str, model, vectorizer) -> tuple[str, int]:
    """Clean, vectorize, and classify a single text input.

    Args:
        text: Raw email text.
        model: Trained classifier.
        vectorizer: Fitted TF-IDF vectorizer.

    Returns:
        Tuple of (label string, raw prediction integer).
    """
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "PHISHING" if prediction == 1 else "SAFE"
    return label, int(prediction)


def load_artifacts(config: dict):
    """Load model and vectorizer from paths in config."""
    try:
        model = joblib.load(config["model"]["best_model_path"])
        vectorizer = joblib.load(config["model"]["vectorizer_path"])
        return model, vectorizer
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model files not found: {e}. Run train.py first."
        )


def main() -> int:
    config = load_config()
    logger = get_logger(__name__, config)

    parser = argparse.ArgumentParser(
        description="Classify email text as PHISHING or SAFE."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text", type=str, help="Email text to classify (quoted string)."
    )
    group.add_argument(
        "--file", type=str, help="Path to a .txt file containing email text."
    )

    args = parser.parse_args()

    try:
        ensure_nltk_data()
        model, vectorizer = load_artifacts(config)

        if args.text:
            raw_text = args.text
        else:
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except FileNotFoundError:
                logger.error(f"File not found: {args.file}")
                return 1

        label, _ = predict(raw_text, model, vectorizer)
        print(f"\nResult: {label}\n")
        logger.info(f"Classified input as: {label}")
        return 0

    except ValueError as e:
        logger.error(f"Prediction failed: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())