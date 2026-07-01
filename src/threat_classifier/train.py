"""Train and evaluate phishing classifiers; save the best model."""
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from threat_classifier.utils import load_config, get_logger

# Hyperparameter grids for tuning
PARAM_GRIDS = {
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]}),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1.0, 10.0]},
    ),
    "SVM": (LinearSVC(), {"C": [0.1, 1.0, 10.0]}),
}


def load_cleaned_data(path: str) -> pd.DataFrame:
    """Load the cleaned dataset, raising clearly if it's missing."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run preprocess.py first."
        )
    df = df.dropna(subset=["clean_text"])
    return df


def train_and_tune(X_train_tfidf, y_train, cv_folds: int, logger) -> dict:
    """Run GridSearchCV for each model and return fitted best estimators.

    Args:
        X_train_tfidf: TF-IDF transformed training features.
        y_train: Training labels.
        cv_folds: Number of stratified CV folds.
        logger: Logger instance.

    Returns:
        Dict mapping model name to its best fitted estimator and CV score.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, (estimator, param_grid) in PARAM_GRIDS.items():
        logger.info(f"Tuning {name} with {cv_folds}-fold CV...")
        search = GridSearchCV(
            estimator, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        search.fit(X_train_tfidf, y_train)
        logger.info(
            f"{name} best params: {search.best_params_}, "
            f"best CV f1: {search.best_score_:.4f}"
        )
        results[name] = {
            "estimator": search.best_estimator_,
            "cv_f1": search.best_score_,
        }

    return results


def main() -> int:
    config = load_config()
    logger = get_logger(__name__, config)
    model_cfg = config["model"]

    try:
        logger.info("Loading cleaned dataset...")
        df = load_cleaned_data(config["data"]["cleaned_path"])

        X = df["clean_text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=model_cfg["test_size"],
            random_state=model_cfg["random_state"],
            stratify=y,
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        vectorizer = TfidfVectorizer(max_features=model_cfg["max_features"])
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        tuned = train_and_tune(
            X_train_tfidf, y_train, model_cfg["cv_folds"], logger
        )

        best_name, best_test_acc, best_model = None, -1, None
        for name, info in tuned.items():
            model = info["estimator"]
            preds = model.predict(X_test_tfidf)
            test_acc = accuracy_score(y_test, preds)
            logger.info(f"\n{name} held-out test accuracy: {test_acc:.4f}")
            logger.info(
                f"\n{classification_report(y_test, preds, target_names=['Safe', 'Phishing'])}"
            )
            logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, preds)}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_name = name
                best_model = model

        logger.info(
            f"\nBest model: {best_name}, held-out test accuracy: {best_test_acc:.4f}"
        )

        joblib.dump(best_model, model_cfg["best_model_path"])
        joblib.dump(vectorizer, model_cfg["vectorizer_path"])
        logger.info(f"Saved model to {model_cfg['best_model_path']}")
        logger.info(f"Saved vectorizer to {model_cfg['vectorizer_path']}")

        return 0
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())