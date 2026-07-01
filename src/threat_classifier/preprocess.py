"""Text preprocessing for the phishing email dataset."""
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from threat_classifier.utils import load_config, get_logger


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation, remove stopwords, and stem.

    Args:
        text: Raw email text.

    Returns:
        Cleaned, stemmed text. Empty string if input isn't a string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def load_and_clean_data(raw_path: str, logger) -> pd.DataFrame:
    """Load the raw CSV, clean text, and map labels to 0/1.

    Args:
        raw_path: Path to the raw CSV file.
        logger: Logger instance for status messages.

    Returns:
        DataFrame with clean_text and label columns added.

    Raises:
        FileNotFoundError: If raw_path doesn't exist.
        ValueError: If expected columns are missing.
    """
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        logger.error(f"Raw dataset not found at {raw_path}")
        raise

    required_cols = {"Email Text", "Email Type"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"Missing expected columns: {missing}")
        raise ValueError(f"Dataset missing columns: {missing}")

    before = len(df)
    df = df[["Email Text", "Email Type"]].dropna()
    dropped = before - len(df)
    logger.info(f"Dropped {dropped} rows with missing values")

    logger.info("Cleaning text (this may take a minute)...")
    df["clean_text"] = df["Email Text"].apply(clean_text)
    df["label"] = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})

    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows had unrecognized Email Type values")

    return df


def main() -> int:
    """Run the full preprocessing pipeline."""
    config = load_config()
    logger = get_logger(__name__, config)

    try:
        ensure_nltk_data()
        df = load_and_clean_data(config["data"]["raw_path"], logger)
        df.to_csv(config["data"]["cleaned_path"], index=False)
        logger.info(f"Saved cleaned dataset to {config['data']['cleaned_path']}")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        return 0
    except Exception as e:
        logger.exception(f"Preprocessing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())