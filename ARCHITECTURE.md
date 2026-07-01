Architecture \& Design Decisions



This document explains how the project is structured, why key technical choices were made, and what tradeoffs exist. Written to support code review and interview discussion.





High-Level Architecture



data/Phishing\_Email.csv  (raw input, 18,650 rows)

&#x20;       |

&#x20;       v

preprocess.py

&#x20; - drops nulls (16 rows removed)

&#x20; - lowercase, strip URLs and punctuation

&#x20; - remove NLTK stopwords, drop tokens <= 2 chars

&#x20; - Porter stemming

&#x20;       |

&#x20;       v

data/cleaned\_dataset.csv  (intermediate artifact, not tracked in git)

&#x20;       |

&#x20;       v

train.py

&#x20; - TF-IDF vectorization (5,000 features)

&#x20; - Stratified 80/20 train/test split

&#x20; - GridSearchCV with 5-fold stratified CV over:

&#x20;       Naive Bayes, Logistic Regression, LinearSVC

&#x20; - Best model selected by held-out test accuracy

&#x20;       |

&#x20;       v

models/best\_model.pkl      (Logistic Regression, C=10)

models/vectorizer.pkl      (fitted TF-IDF vectorizer)

&#x20;       |

&#x20;       v

Two interfaces:

&#x20; app.py        Streamlit web UI (analyst-facing)

&#x20; predict.py    CLI (scriptable, pipeline-friendly)





Why TF-IDF + Classical ML, Not a Transformer?



This is the most common question this project invites, so it deserves a direct answer.



TF-IDF + linear models were chosen deliberately, not by default.



1\. The task fits the approach.

Phishing emails rely heavily on specific high-signal vocabulary — "urgent", "verify", "suspend", "password", "click". TF-IDF weights exactly these rare-but-distinctive terms highly. A bag-of-words representation captures the core signal without needing contextual embeddings.



2\. Results validate the choice.

96.75% accuracy on a held-out test set of 3,726 samples is strong performance. A transformer would likely improve this marginally — perhaps 1-2% — at significant cost in training time, inference latency, and hardware requirements.



3\. Interpretability matters in a SOC context.

Linear models let you inspect feature weights directly — you can see which words drove a phishing classification. This is operationally useful: an analyst can verify the reasoning, not just trust a black box.



4\. Deployment simplicity.

The full model + vectorizer is \~5MB and classifies in milliseconds. A transformer would require GPU inference or significant latency on CPU — impractical for real-time SOAR integration.



The honest tradeoff: TF-IDF won't catch sophisticated phishing that avoids flagged vocabulary, uses image-based content, or exploits sender/header manipulation. A production system would need additional signal sources beyond email body text.





Why Logistic Regression Beat SVM After Tuning?



The untuned baseline had SVM slightly ahead (96.62% vs 96.43%). After GridSearchCV:





LR with C=10 reached 96.75%

SVM with C=1.0 reached 96.62%





The likely explanation: at default C=1, LR was slightly under-fitting. Increasing C allowed it to fit the decision boundary more tightly on TF-IDF features, which are sparse and high-dimensional — a setting where LR often performs competitively with SVM. The key lesson: do not assume which model wins without tuning both.





Package Structure Rationale



nlp-threat-classifier/

├── src/

│   └── threat\_classifier/

│       ├── \_\_init\_\_.py

│       ├── utils.py          shared: config loading, logger factory

│       ├── preprocess.py     text cleaning (used by train, app, predict, tests)

│       ├── train.py          training pipeline

│       ├── app.py            Streamlit UI

│       └── predict.py        CLI interface

├── tests/                    pytest unit tests, separate from src

├── config.yaml               all paths and hyperparameters in one place

└── setup.py                  makes the package pip-installable



Why a package instead of flat scripts?

Flat scripts work but create import problems — every file has to use fragile relative path hacks to share code. Making it a proper package means from threat\_classifier.preprocess import clean\_text works cleanly from anywhere, including tests, without sys.path manipulation. It also reflects how real Python projects are structured.



Why config.yaml?

All paths, model hyperparameters, and CV settings live in one file rather than being hardcoded across three scripts. Changing max\_features or cv\_folds means editing one line in one file, not hunting through code. In a production system this would extend to environment-specific configs (dev/staging/prod).





What Is Not Here and Why



MissingReasonConfidence scores in outputLinearSVC does not produce calibrated probabilities natively; requires CalibratedClassifierCV wrapper — left as a next stepSOAR/SIEM integrationOut of scope for a portfolio project; predict.py is the integration point a SOAR platform would callAdversarial robustness testingNot validated against evasion attempts; a real deployment would need red-team testingTransformer baselineDeliberately excluded — see TF-IDF rationale aboveDocker verified buildDockerfile is written but not yet confirmed to build end-to-end on this machine

