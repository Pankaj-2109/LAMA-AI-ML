import joblib
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INTENT_FRIENDLY_NAMES = {
    "technical_coding": "Technical & Coding",
    "conceptual_educational": "Conceptual & Educational",
    "conversational_casual": "Conversational & Casual",
    "problem_solving_advice": "Problem Solving & Advice",
    "creative_brainstorming": "Creative & Brainstorming",
    "factual_inquiry": "Factual & Direct Inquiry"
}

class IntentClassifier:
    """Predicts user intent using TF-IDF features."""

    def __init__(self, alpha: float = 0.1):
        self.model = MultinomialNB(alpha=alpha)
        self.classes_ = []
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self

    def predict(self, feature_vector):
        """
        Takes a sparse TF-IDF feature matrix (1 x N).
        Returns predicted intent, friendly label, and confidence score.
        """
        if not self.is_fitted:
            raise ValueError("IntentClassifier has not been fitted yet.")

        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes_, probabilities)
        }
        confidence = prob_dict.get(prediction, 0.5)

        return {
            "intent": prediction,
            "friendly_name": INTENT_FRIENDLY_NAMES.get(prediction, prediction),
            "confidence": round(float(confidence), 3),
            "probabilities": prob_dict
        }

    def evaluate(self, X_test, y_test, X_all=None, y_all=None):
        """Returns evaluation dictionary for metrics logging."""
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()

        cv_acc = acc
        if X_all is not None and y_all is not None:
            try:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(self.model, X_all, y_all, cv=5)
                cv_acc = float(np.mean(scores))
            except Exception:
                pass

        return {
            "accuracy": round(float(acc), 4),
            "cv_accuracy": round(float(cv_acc), 4),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "classes": self.classes_
        }

    def save(self, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes_}, file_path)

    def load(self, file_path: str):
        data = joblib.load(file_path)
        self.model = data["model"]
        self.classes_ = data["classes"]
        self.is_fitted = True
        return self
