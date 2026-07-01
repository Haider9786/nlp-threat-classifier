FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

# Pre-download NLTK data at build time so the container doesn't need
# internet access at runtime
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

EXPOSE 8501

CMD ["streamlit", "run", "src/threat_classifier/app.py", "--server.address=0.0.0.0"]