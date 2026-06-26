# NLP Threat Intelligence Text Classifier

**Stack:** Python | Scikit-learn | NLTK | TF-IDF | Streamlit  
**Domain:** Threat Intelligence | SOC Automation | NLP/ML  
**Type:** Final Year Project — University of Chakwal (2025)

---

## Project Overview

This project builds a machine learning classification pipeline that automatically categorises threat intelligence text into meaningful security categories. The motivation was a real SOC problem analysts spend significant time manually triaging unstructured threat intelligence feeds, phishing reports, and security bulletins. This classifier automates that triage step.

The system processes raw threat intelligence text and classifies it into categories including phishing indicators, malware descriptions, network-based threats, and benign content. It was trained and evaluated on a dataset of 1,000+ labelled threat intelligence samples.

---

## The SOC Problem This Solves

In a real SOC environment, analysts receive threat intelligence from multiple feeds commercial threat intel platforms, open source feeds, email reports, and incident sharing communities. Each entry needs to be read, understood, and categorised before it can be actioned. For a busy SOC processing hundreds of entries daily, this is a significant manual workload.

This classifier addresses that by:
- Automatically tagging incoming threat intel by category
- Flagging high-priority entries (phishing, active malware) for immediate analyst review
- Filtering benign or low-confidence entries to reduce noise
- Providing a foundation for SOAR integration and automated ticket creation

---

## Architecture

```
Raw Threat Intelligence Text
          |
          v
+------------------------+
|   Text Preprocessing   |
|   - Tokenisation       |
|   - Stop word removal  |
|   - Stemming (NLTK)    |
+------------------------+
          |
          v
+------------------------+
|   Feature Extraction   |
|   - TF-IDF Vectors     |
|   - Bag of Words       |
+------------------------+
          |
          v
+------------------------+
|   ML Classification    |
|   - Naive Bayes        |
|   - Logistic Regression|
|   - SVM                |
+------------------------+
          |
          v
+------------------------+
|   Streamlit Dashboard  |
|   - Input text field   |
|   - Category output    |
|   - Confidence score   |
+------------------------+
```

---

## Dataset

- **Size:** 1,000+ labelled threat intelligence samples
- **Sources:** Open source threat intelligence reports, CVE descriptions, phishing alert examples, security bulletins
- **Categories:**
  - Phishing:Email-based social engineering indicators
  - Malware:Malicious software descriptions and IoCs
  - Network Threat:Network-based attack indicators
  - Vulnerability:CVE and exploit descriptions
  - Benign:Non-threatening security content

---

## Technical Implementation

### Preprocessing Pipeline

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    # Tokenise
    tokens = nltk.word_tokenize(text.lower())
    # Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Stem
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)
```

### Feature Extraction

Two approaches were implemented and compared:

**TF-IDF (Term Frequency-Inverse Document Frequency)**
Weights terms by how often they appear in a document relative to the whole corpus. Cybersecurity-specific terms like "exfiltration", "payload", "C2", and "lateral" score highly, making TF-IDF well-suited for threat intelligence text.

**Bag of Words**
Simple word count vectorisation. Used as a baseline comparison against TF-IDF.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes + TF-IDF | 84% | 0.83 | 0.84 | 0.83 |
| Logistic Regression + TF-IDF | 88% | 0.87 | 0.88 | 0.87 |
| SVM + TF-IDF | 91% | 0.90 | 0.91 | 0.90 |
| Naive Bayes + BoW | 79% | 0.78 | 0.79 | 0.78 |

SVM with TF-IDF achieved the best overall performance at 91% accuracy.

---

## Streamlit Dashboard

The classifier is wrapped in a Streamlit web application allowing analysts to paste threat intelligence text and receive an instant classification with confidence score.

**Features:**
- Text input field for raw threat intelligence
- Category prediction with confidence percentage
- Colour-coded output:red for high-priority threats, amber for medium, green for benign
- Batch processing mode for CSV upload of multiple entries

---

## SOC Integration Potential

This classifier was designed with real SOC integration in mind:

**SOAR Integration**
The model can be wrapped as an API endpoint and called by SOAR platforms to automatically tag incoming threat intelligence before it reaches the analyst queue.

**Splunk/Wazuh Integration**
Output classifications can be fed back into SIEM platforms as enrichment fields on alerts, providing additional context during triage.

**Threat Feed Automation**
When connected to open source threat feeds like AlienVault OTX or MISP, the classifier can pre-process incoming IoCs and route them to the relevant analyst team automatically.

---

## Files

```
nlp-threat-classifier/
├── README.md
├── data/
│   ├── threat_intel_dataset.csv
│   └── labels.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── app.py (Streamlit)
├── models/
│   └── svm_tfidf_model.pkl
└── requirements.txt
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/shahabminhas/nlp-threat-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run src/app.py
```

---

## Lessons Learned

**1. Domain-specific preprocessing matters**
Generic stop word lists remove words like "not" which are semantically important in threat descriptions. Custom preprocessing that retains negations improved accuracy by approximately 4%.

**2. TF-IDF outperforms Bag of Words for threat intel**
Because threat intelligence text is highly specialised, rare domain-specific terms carry disproportionate signal value. TF-IDF captures this; Bag of Words treats all terms equally.

**3. Class imbalance is a real challenge**
Phishing samples significantly outnumbered other categories in the initial dataset. SMOTE oversampling was applied to balance the training set, which improved recall on underrepresented categories.

**4. The bridge between ML and security is underexplored**
Most cybersecurity practitioners are not machine learning engineers and most ML engineers do not understand threat intelligence workflows. Building this project gave me a perspective that sits at the intersection of both directly applicable to SOAR automation, anomaly detection, and AI-assisted SOC operations.
