"""
Step 2 & Step 3: Text Preprocessor and TF-IDF Feature Extractor for LAMA AI.
"""

import re
import string
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Basic standard English stopwords list (self-contained to prevent runtime downloads)
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}

class TextPreprocessor:
    """Step 2: Preprocess user input text."""

    @staticmethod
    def clean_text(text: str, remove_stopwords: bool = False) -> str:
        if not text or not isinstance(text, str):
            return ""

        # Lowercase
        cleaned = text.lower().strip()

        # Remove URLs
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)

        # Remove HTML tags
        cleaned = re.sub(r"<.*?>", "", cleaned)

        # Strip non-alphanumeric punctuation except code symbols useful for intent classification
        # Retain tokens like c++, c#, .js, etc.
        cleaned = re.sub(r"[^\w\s\+\#\.]", " ", cleaned)

        # Tokenize and remove extra whitespace
        tokens = cleaned.split()

        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]

        return " ".join(tokens)


class TFIDFExtractor:
    """Step 3: Extract Text Features Using TF-IDF."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            stop_words="english",
            strip_accents="unicode"
        )
        self.is_fitted = False

    def fit(self, raw_documents):
        cleaned_docs = [TextPreprocessor.clean_text(doc) for doc in raw_documents]
        self.vectorizer.fit(cleaned_docs)
        self.is_fitted = True
        return self

    def transform(self, raw_documents):
        if not self.is_fitted:
            raise ValueError("TFIDFExtractor has not been fitted yet.")
        if isinstance(raw_documents, str):
            raw_documents = [raw_documents]
        cleaned_docs = [TextPreprocessor.clean_text(doc) for doc in raw_documents]
        return self.vectorizer.transform(cleaned_docs)

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def save(self, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, file_path)

    def load(self, file_path: str):
        self.vectorizer = joblib.load(file_path)
        self.is_fitted = True
        return self
