"""
Step 10: Response Quality Predictor for LAMA AI.
Extracts cross-features between prompt and generated response to predict quality score (0.0 - 1.0).
"""

import re
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class QualityPredictor:
    """Predicts the quality match of a generated response for a given prompt and user persona."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha, random_state=42)
        self.is_fitted = False

    @staticmethod
    def extract_features(prompt: str, response: str, cluster_id: int = 0) -> list:
        """
        Extracts 6 predictive features:
        1. Word overlap / Jaccard similarity
        2. Response length score (relative to persona preference)
        3. Structure richness (markdown, code blocks, bullet points)
        4. Vocabulary diversity (type-token ratio)
        5. Sentence count & completeness
        6. Persona alignment penalty/reward
        """
        p_clean = re.sub(r"[^\w\s]", "", (prompt or "").lower()).split()
        r_clean = re.sub(r"[^\w\s]", "", (response or "").lower()).split()

        p_words = set(p_clean)
        r_words = set(r_clean)

        # 1. Jaccard similarity
        intersection = len(p_words & r_words)
        union = len(p_words | r_words) if p_words or r_words else 1
        jaccard = intersection / max(union, 1)

        # 2. Length score based on cluster persona
        # Cluster 0 (Pragmatist): Ideal 30-100 words
        # Cluster 1 (Deep-Diver): Ideal 150-500 words
        # Cluster 2 (Conversationalist): Ideal 50-200 words
        # Cluster 3 (Brainstormer): Ideal 100-300 words
        ideal_lengths = {0: (30, 100), 1: (150, 500), 2: (50, 200), 3: (100, 300)}
        min_len, max_len = ideal_lengths.get(cluster_id, (50, 250))
        r_len = len(r_clean)

        if r_len < min_len:
            len_score = max(0.2, r_len / max(min_len, 1))
        elif r_len <= max_len:
            len_score = 1.0
        else:
            len_score = max(0.5, 1.0 - (r_len - max_len) / 500.0)

        # 3. Structure score (code blocks, headers, bullet points)
        structure_score = 0.5
        if "```" in response:
            structure_score += 0.2
        if any(line.strip().startswith(("-", "*", "1.", "2.")) for line in response.split("\n")):
            structure_score += 0.15
        if "#" in response:
            structure_score += 0.15
        structure_score = min(1.0, structure_score)

        # 4. Vocabulary diversity
        diversity = len(r_words) / max(len(r_clean), 1) if r_clean else 0.5

        # 5. Non-empty completeness
        completeness = 1.0 if len(response.strip()) > 10 else 0.1

        # 6. Cluster match flag
        cluster_factor = 0.8 + (cluster_id * 0.05)

        return [
            round(float(jaccard), 4),
            round(float(len_score), 4),
            round(float(structure_score), 4),
            round(float(diversity), 4),
            round(float(completeness), 4),
            round(float(cluster_factor), 4)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, prompt: str, response: str, cluster_id: int = 0) -> dict:
        features = self.extract_features(prompt, response, cluster_id)
        if not self.is_fitted:
            # Heuristic default based on extracted features
            raw_score = 0.3 * features[0] + 0.3 * features[1] + 0.2 * features[2] + 0.2 * features[3]
            score = float(np.clip(raw_score * 1.2, 0.65, 0.98))
        else:
            arr = np.array(features).reshape(1, -1)
            raw_score = float(self.model.predict(arr)[0])
            score = float(np.clip(raw_score, 0.50, 0.99))

        pct = int(round(score * 100))
        return {
            "quality_score": round(score, 3),
            "percentage": pct,
            "grade": "Excellent" if pct >= 90 else "Good" if pct >= 75 else "Adequate",
            "features": {
                "jaccard_overlap": features[0],
                "length_appropriateness": features[1],
                "structural_richness": features[2],
                "vocabulary_diversity": features[3]
            }
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = np.clip(self.model.predict(X_test), 0.0, 1.0)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        return {
            "mse": round(float(mse), 5),
            "mae": round(float(mae), 4),
            "r2_score": round(float(r2), 4)
        }

    def save(self, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, file_path)

    def load(self, file_path: str):
        self.model = joblib.load(file_path)
        self.is_fitted = True
        return self
