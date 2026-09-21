"""
Step 15: Training, Evaluation, and Serialization Pipeline for LAMA AI ML Models.
Executes training for:
1. TF-IDF + Intent Classifier
2. K-Means User Profiler & Clusterer
3. Response Quality Regressor
Computes and persists evaluation metrics (Accuracy, F1, Silhouette score, MSE, R2).
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from data.seed_data import INTENT_DATA, SYNTHETIC_USER_PROFILES
from preprocessor import TFIDFExtractor
from intent_classifier import IntentClassifier
from user_profiler import UserProfiler
from quality_predictor import QualityPredictor

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

def train_and_evaluate(extra_intent_samples=None, extra_profiles=None, extra_quality_samples=None):
    """Executes the full ML training and evaluation cycle."""
    print("[TRAINER] Starting LAMA AI Model Training & Evaluation Pipeline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # =========================================================================
    # 1. TRAIN & EVALUATE TF-IDF + INTENT CLASSIFIER (Step 3 & 4)
    # =========================================================================
    print("-> Training TF-IDF & Intent Classifier...")
    all_intent_data = list(INTENT_DATA)
    if extra_intent_samples:
        all_intent_data.extend(extra_intent_samples)

    prompts = [item[0] for item in all_intent_data]
    labels = [item[1] for item in all_intent_data]

    # Split for train & test evaluation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        prompts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    tfidf = TFIDFExtractor()
    X_train_vec = tfidf.fit_transform(X_train_raw)
    X_test_vec = tfidf.transform(X_test_raw)

    classifier = IntentClassifier()
    classifier.fit(X_train_vec, y_train)

    X_all_vec = tfidf.transform(prompts)
    intent_metrics = classifier.evaluate(X_test_vec, y_test, X_all=X_all_vec, y_all=labels)
    print(f"   Intent Model Accuracy: {intent_metrics['accuracy'] * 100:.2f}% | CV Accuracy: {intent_metrics['cv_accuracy'] * 100:.2f}% | Macro F1: {intent_metrics['macro_f1']:.4f}")

    # Refit on full dataset for maximum production capability
    tfidf_prod = TFIDFExtractor()
    X_full_vec = tfidf_prod.fit_transform(prompts)
    classifier_prod = IntentClassifier()
    classifier_prod.fit(X_full_vec, labels)

    # Save models
    tfidf_prod.save(str(MODELS_DIR / "tfidf_vectorizer.joblib"))
    classifier_prod.save(str(MODELS_DIR / "intent_classifier.joblib"))

    # =========================================================================
    # 2. TRAIN & EVALUATE K-MEANS USER CLUSTERING (Step 7)
    # =========================================================================
    print("-> Training K-Means User Profiler & Clusterer...")
    all_profiles = list(SYNTHETIC_USER_PROFILES)
    if extra_profiles:
        all_profiles.extend(extra_profiles)

    profiles_matrix = np.array(all_profiles)
    profiler = UserProfiler(n_clusters=4)
    profiler.fit(profiles_matrix)
    clustering_metrics = profiler.evaluate(profiles_matrix)
    print(f"   K-Means Silhouette Score: {clustering_metrics['silhouette_score']:.4f} | Inertia: {clustering_metrics['inertia']}")

    profiler.save(str(MODELS_DIR / "user_profiler.joblib"))

    # =========================================================================
    # 3. TRAIN & EVALUATE RESPONSE QUALITY REGRESSOR (Step 10)
    # =========================================================================
    print("-> Training Response Quality Predictor...")
    # Synthetic feature dataset for quality regression
    # Features: [jaccard, len_score, structure_score, diversity, completeness, cluster_factor]
    np.random.seed(42)
    n_quality_samples = 150
    X_qual = []
    y_qual = []

    for _ in range(n_quality_samples):
        jaccard = np.random.uniform(0.1, 0.7)
        len_score = np.random.uniform(0.4, 1.0)
        structure_score = np.random.uniform(0.4, 1.0)
        diversity = np.random.uniform(0.5, 0.9)
        completeness = 1.0
        cluster_factor = np.random.uniform(0.8, 1.0)

        # Ground truth target: weighted formula + slight noise
        target = (
            0.25 * jaccard +
            0.30 * len_score +
            0.25 * structure_score +
            0.15 * diversity +
            0.05 * cluster_factor +
            np.random.normal(0, 0.03)
        )
        target = float(np.clip(target, 0.60, 0.98))

        X_qual.append([jaccard, len_score, structure_score, diversity, completeness, cluster_factor])
        y_qual.append(target)

    X_qual_arr = np.array(X_qual)
    y_qual_arr = np.array(y_qual)

    X_q_train, X_q_test, y_q_train, y_q_test = train_test_split(
        X_qual_arr, y_qual_arr, test_size=0.2, random_state=42
    )

    quality_pred = QualityPredictor()
    quality_pred.fit(X_q_train, y_q_train)
    quality_metrics = quality_pred.evaluate(X_q_test, y_q_test)
    print(f"   Quality Predictor R2: {quality_metrics['r2_score']:.4f} | MSE: {quality_metrics['mse']:.5f}")

    quality_pred.save(str(MODELS_DIR / "quality_predictor.joblib"))

    # =========================================================================
    # 4. SAVE COMPREHENSIVE METRICS REPORT (Step 15 output)
    # =========================================================================
    duration = round(time.time() - start_time, 2)
    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration,
        "intent_classification": intent_metrics,
        "user_clustering": clustering_metrics,
        "response_quality_prediction": quality_metrics,
        "dataset_summary": {
            "intent_samples": len(all_intent_data),
            "user_profile_samples": len(all_profiles),
            "quality_samples": len(X_qual)
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"[DONE] Training completed successfully in {duration}s! Metrics saved to {METRICS_PATH}")
    return final_report

if __name__ == "__main__":
    train_and_evaluate()
