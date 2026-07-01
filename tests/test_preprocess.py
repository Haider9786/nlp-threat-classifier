"""Unit tests for the preprocessing module."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from threat_classifier.preprocess import clean_text


def test_clean_text_lowercases():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_removes_urls():
    result = clean_text("Click here http://evil.com now")
    assert "http" not in result
    assert "evil" not in result


def test_clean_text_removes_punctuation_and_numbers():
    result = clean_text("Win $1000 now!!! Call 555-1234")
    assert "1000" not in result
    assert "555" not in result
    assert "!" not in result


def test_clean_text_removes_stopwords():
    result = clean_text("this is a test of the system")
    # "this", "is", "a", "of", "the" are stopwords and should be gone
    assert "this" not in result.split()
    assert "the" not in result.split()


def test_clean_text_removes_short_words():
    # words with len <= 2 are dropped
    result = clean_text("hi to me at it")
    assert result.strip() == ""


def test_clean_text_stems_words():
    result = clean_text("running runner runs")
    # PorterStemmer should reduce these to a common root containing "run"
    assert all("run" in token for token in result.split())


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


def test_clean_text_phishing_example():
    result = clean_text("URGENT: verify your password immediately!")
    assert "urgent" in result
    assert "verifi" in result  # stemmed form of "verify"
    assert "password" in result