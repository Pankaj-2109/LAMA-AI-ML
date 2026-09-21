"""
LAMA AI ML Service: Flask REST API on port 5001.
Coordinates Step 1 through Step 15 of the Machine Learning Pipeline.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocessor import TextPreprocessor, TFIDFExtractor
from intent_classifier import IntentClassifier
from user_profiler import UserProfiler, CLUSTER_DEFINITIONS
from strategy_selector import StrategySelector
from quality_predictor import QualityPredictor
from trainer import train_and_evaluate, MODELS_DIR, METRICS_PATH

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================================================================
# LOAD MODELS AT STARTUP
# =========================================================================
tfidf = None
intent_model = None
profiler = None
quality_model = None

def load_models():
    global tfidf, intent_model, profiler, quality_model
    print("[SERVICE] Loading ML model artifacts...")

    tfidf_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    intent_path = MODELS_DIR / "intent_classifier.joblib"
    profiler_path = MODELS_DIR / "user_profiler.joblib"
    quality_path = MODELS_DIR / "quality_predictor.joblib"

    if not tfidf_path.exists() or not intent_path.exists() or not profiler_path.exists():
        print("[SERVICE] Models not found. Initiating first-time training...")
        train_and_evaluate()

    tfidf = TFIDFExtractor().load(str(tfidf_path))
    intent_model = IntentClassifier().load(str(intent_path))
    profiler = UserProfiler().load(str(profiler_path))
    quality_model = QualityPredictor().load(str(quality_path))
    print("[SERVICE] All ML models loaded successfully into memory.")

load_models()

# =========================================================================
# ROUTES
# =========================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "LAMA AI ML Engine",
        "version": "1.0.0",
        "models_loaded": {
            "tfidf": tfidf is not None and tfidf.is_fitted,
            "intent_classifier": intent_model is not None and intent_model.is_fitted,
            "user_profiler": profiler is not None and profiler.is_fitted,
            "quality_predictor": quality_model is not None and quality_model.is_fitted
        }
    })

@app.route("/ml/pipeline/pre-generation", methods=["POST"])
def pre_generation():
    """
    Executes Steps 2 through 8 of the LAMA AI pipeline:
    - Step 2: Preprocess Prompt
    - Step 3: Extract TF-IDF features
    - Step 4: Predict Intent
    - Step 5 & 6: Create User Behaviour Profile from chat history
    - Step 7: Assign User to Cluster using K-Means
    - Step 8: Select Personalized Strategy & System Instruction for Gemini
    """
    try:
        data = request.get_json() or {}
        raw_prompt = data.get("prompt", "")
        history = data.get("history", [])
        feedback_history = data.get("feedback_history", [])
        custom_vector = data.get("user_profile_vector", None)

        if not raw_prompt.strip():
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Step 2: Preprocess the Prompt
        cleaned_prompt = TextPreprocessor.clean_text(raw_prompt)

        # Step 3: Extract Text Features Using TF-IDF
        feature_vec = tfidf.transform(cleaned_prompt)

        # Step 4: Predict the Prompt's Intent
        intent_result = intent_model.predict(feature_vec)

        # Step 5 & 6: Create User's Behaviour Profile
        if custom_vector and isinstance(custom_vector, list) and len(custom_vector) == 6:
            profile_vector = custom_vector
        else:
            profile_vector = profiler.extract_profile_from_history(history, feedback_history)

        # Step 7: Assign User to Cluster using K-Means
        cluster_result = profiler.assign_cluster(profile_vector)

        # Step 8: Select a Personalized Response Strategy
        strategy_result = StrategySelector.select_strategy(
            intent_data=intent_result,
            cluster_data=cluster_result,
            profile_vector=profile_vector
        )

        return jsonify({
            "status": "success",
            "preprocessed_prompt": cleaned_prompt,
            "intent": intent_result,
            "user_profile": {
                "vector": profile_vector,
                "labels": ["avg_prompt_len", "code_ratio", "curiosity_depth", "sentiment_tone", "avg_feedback", "conciseness_pref"]
            },
            "cluster": cluster_result,
            "strategy": strategy_result
        })

    except Exception as e:
        print("[SERVICE ERROR] Pre-generation error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/ml/predict-quality", methods=["POST"])
def predict_quality():
    """
    Executes Step 10: Predicts response quality based on prompt, response, and persona.
    """
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "")
        response = data.get("response", "")
        cluster_id = int(data.get("cluster_id", 0))

        prediction = quality_model.predict(prompt, response, cluster_id)

        return jsonify({
            "status": "success",
            "prediction": prediction
        })
    except Exception as e:
        print("[SERVICE ERROR] Quality prediction error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/ml/update-profile", methods=["POST"])
def update_profile():
    """
    Executes Step 14: Updates user behavioral profile with feedback via online learning.
    """
    try:
        data = request.get_json() or {}
        current_vector = data.get("current_vector", [40.0, 0.25, 0.4, 0.6, 4.0, 0.5])
        feedback = data.get("feedback", {})

        updated_vector = UserProfiler.update_profile_with_feedback(current_vector, feedback)
        new_cluster = profiler.assign_cluster(updated_vector)

        return jsonify({
            "status": "success",
            "updated_vector": updated_vector,
            "new_cluster": new_cluster
        })
    except Exception as e:
        print("[SERVICE ERROR] Profile update error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/ml/retrain", methods=["POST"])
def retrain():
    """
    Executes Step 15: Retrains and evaluates the ML models.
    """
    try:
        data = request.get_json() or {}
        extra_intents = data.get("extra_intents", None)
        extra_profiles = data.get("extra_profiles", None)

        report = train_and_evaluate(
            extra_intent_samples=extra_intents,
            extra_profiles=extra_profiles
        )

        # Reload updated models
        load_models()

        return jsonify({
            "status": "success",
            "message": "Models retrained and reloaded successfully",
            "metrics": report
        })
    except Exception as e:
        print("[SERVICE ERROR] Retrain error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/ml/metrics", methods=["GET"])
def get_metrics():
    """
    Returns the latest model evaluation metrics.
    """
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({"error": "Metrics file not found. Please train models first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ml/cluster-definitions", methods=["GET"])
def get_clusters():
    """Returns cluster personas and definitions."""
    return jsonify(CLUSTER_DEFINITIONS)

if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 5001))
    print(f"[SERVICE] Starting LAMA AI ML Service on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
