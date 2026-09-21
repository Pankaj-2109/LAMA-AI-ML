"""
Step 5, 6, 7 & 14: User Behaviour Profiler, K-Means Clustering, and Online Preference Updates.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

CLUSTER_DEFINITIONS = {
    0: {
        "name": "The Pragmatist",
        "description": "Prefers quick, direct answers, concise bullet points, and actionable summaries without filler.",
        "badge_emoji": "🎯",
        "style_directive": "Keep responses concise, crisp, and direct. Use bullet points and TL;DR summaries. Avoid introductory fluff."
    },
    1: {
        "name": "The Deep-Diver",
        "description": "Prefers thorough technical explanations, architectural context, edge cases, and robust code snippets.",
        "badge_emoji": "⚡",
        "style_directive": "Provide comprehensive, deeply technical, structured answers with code examples, underlying mechanics, and best practices."
    },
    2: {
        "name": "The Conversationalist",
        "description": "Prefers warm, empathetic, engaging dialogue with relatable examples and friendly follow-ups.",
        "badge_emoji": "💬",
        "style_directive": "Maintain an engaging, friendly, and conversational tone. Use analogies and offer helpful, empathetic follow-up suggestions."
    },
    3: {
        "name": "The Brainstormer",
        "description": "Prefers creative ideation, innovative alternatives, multiple perspectives, and structured brainstorming.",
        "badge_emoji": "💡",
        "style_directive": "Offer multi-faceted creative ideas, diverse perspectives, pros/cons, and out-of-the-box suggestions."
    }
}

class UserProfiler:
    """Extracts behavioral vectors, clusters users via K-Means, and updates preferences."""

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.is_fitted = False

    def fit(self, profiles_matrix: np.ndarray):
        """Fit scaler and KMeans on synthetic/historical profile feature matrix."""
        scaled_features = self.scaler.fit_transform(profiles_matrix)
        self.kmeans.fit(scaled_features)
        self.is_fitted = True
        return self

    @staticmethod
    def extract_profile_from_history(history_list: list, feedback_list: list = None) -> list:
        """
        Step 5 & 6: Builds a 6D behavioral vector from user's chat history and feedback:
        [avg_prompt_len, code_query_ratio, curiosity_depth, sentiment_tone, avg_feedback, conciseness_pref]
        """
        if not history_list:
            # Default cold-start feature vector
            return [40.0, 0.25, 0.4, 0.6, 4.0, 0.5]

        user_prompts = []
        code_keywords = {"code", "function", "debug", "error", "api", "react", "python", "js", "sql", "git", "bug", "class"}
        curiosity_keywords = {"why", "how", "explain", "what", "compare", "difference", "understand"}

        code_count = 0
        curiosity_count = 0

        for item in history_list:
            text = ""
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                # Standard chat part or message
                parts = item.get("parts", [])
                if parts and isinstance(parts[0], dict):
                    text = parts[0].get("text", "")
                elif "text" in item:
                    text = item["text"]
                elif "question" in item:
                    text = item["question"] or ""

            if text.strip():
                user_prompts.append(text)
                words = set(text.lower().split())
                if words & code_keywords:
                    code_count += 1
                if words & curiosity_keywords or "?" in text:
                    curiosity_count += 1

        total_prompts = max(len(user_prompts), 1)
        avg_len = sum(len(p) for p in user_prompts) / total_prompts
        code_ratio = code_count / total_prompts
        curiosity_depth = min(curiosity_count / total_prompts, 1.0)

        # Tone / sentiment heuristic (defaults to neutral-positive 0.6)
        sentiment_tone = 0.6

        # Average feedback
        if feedback_list and len(feedback_list) > 0:
            valid_ratings = [f.get("rating", 4) for f in feedback_list if isinstance(f, dict) and "rating" in f]
            avg_feedback = sum(valid_ratings) / max(len(valid_ratings), 1) if valid_ratings else 4.0
        else:
            avg_feedback = 4.0

        # Conciseness preference: based on length preference tags or feedback
        conciseness_pref = 0.5
        if feedback_list:
            concise_votes = sum(1 for f in feedback_list if "Concise" in f.get("tags", []))
            detailed_votes = sum(1 for f in feedback_list if "Detailed" in f.get("tags", []))
            total_votes = concise_votes + detailed_votes
            if total_votes > 0:
                conciseness_pref = concise_votes / total_votes

        return [
            round(float(avg_len), 2),
            round(float(code_ratio), 3),
            round(float(curiosity_depth), 3),
            round(float(sentiment_tone), 3),
            round(float(avg_feedback), 2),
            round(float(conciseness_pref), 3)
        ]

    def assign_cluster(self, profile_vector: list) -> dict:
        """
        Step 7: Scale profile and predict K-Means cluster.
        Returns cluster index, persona name, and style directive.
        """
        if not self.is_fitted:
            # Fallback if not fitted
            return {
                "cluster_id": 0,
                "name": CLUSTER_DEFINITIONS[0]["name"],
                "badge_emoji": CLUSTER_DEFINITIONS[0]["badge_emoji"],
                "description": CLUSTER_DEFINITIONS[0]["description"],
                "style_directive": CLUSTER_DEFINITIONS[0]["style_directive"]
            }

        arr = np.array(profile_vector).reshape(1, -1)
        scaled = self.scaler.transform(arr)
        cluster_id = int(self.kmeans.predict(scaled)[0])

        cluster_info = CLUSTER_DEFINITIONS.get(cluster_id, CLUSTER_DEFINITIONS[0])
        return {
            "cluster_id": cluster_id,
            "name": cluster_info["name"],
            "badge_emoji": cluster_info["badge_emoji"],
            "description": cluster_info["description"],
            "style_directive": cluster_info["style_directive"]
        }

    @staticmethod
    def update_profile_with_feedback(current_vector: list, feedback: dict, alpha: float = 0.2) -> list:
        """
        Step 14: Update User's Preferences via Exponential Moving Average.
        - alpha is the learning rate / adaptation factor.
        """
        updated = list(current_vector)
        rating = float(feedback.get("rating", 4))
        # Update avg feedback (feature 4)
        updated[4] = round((1 - alpha) * updated[4] + alpha * rating, 2)

        tags = feedback.get("tags", [])
        if "Too long" in tags or "Concise" in tags:
            # User wants shorter responses -> increase conciseness preference (feature 5)
            updated[5] = round(min(1.0, updated[5] + 0.1), 3)
        elif "Detailed" in tags or "Needs more detail" in tags:
            # User wants more depth -> decrease conciseness preference
            updated[5] = round(max(0.0, updated[5] - 0.1), 3)

        return updated

    def evaluate(self, profiles_matrix: np.ndarray) -> dict:
        """Evaluates clustering using Silhouette score and Inertia."""
        scaled = self.scaler.transform(profiles_matrix)
        labels = self.kmeans.predict(scaled)
        try:
            sil_score = float(silhouette_score(scaled, labels))
        except Exception:
            sil_score = 0.5

        return {
            "n_clusters": self.n_clusters,
            "inertia": round(float(self.kmeans.inertia_), 2),
            "silhouette_score": round(sil_score, 4),
            "cluster_distribution": {
                int(c): int(np.sum(labels == c)) for c in range(self.n_clusters)
            }
        }

    def save(self, file_path: str):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "kmeans": self.kmeans}, file_path)

    def load(self, file_path: str):
        data = joblib.load(file_path)
        self.scaler = data["scaler"]
        self.kmeans = data["kmeans"]
        self.is_fitted = True
        return self
