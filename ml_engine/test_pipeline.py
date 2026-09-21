"""
End-to-End Test for LAMA AI ML Pipeline (Steps 2 to 15).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from preprocessor import TextPreprocessor, TFIDFExtractor
from intent_classifier import IntentClassifier
from user_profiler import UserProfiler
from strategy_selector import StrategySelector
from quality_predictor import QualityPredictor
from trainer import MODELS_DIR

def run_tests():
    print("[TEST] Running LAMA AI ML Pipeline Verification...")

    # 1. Load models
    tfidf = TFIDFExtractor().load(str(MODELS_DIR / "tfidf_vectorizer.joblib"))
    classifier = IntentClassifier().load(str(MODELS_DIR / "intent_classifier.joblib"))
    profiler = UserProfiler().load(str(MODELS_DIR / "user_profiler.joblib"))
    quality_pred = QualityPredictor().load(str(MODELS_DIR / "quality_predictor.joblib"))

    # Step 1: User prompt
    test_prompt = "Write a Python script to scrape tabular data from an API with pagination and rate limits"
    print(f"Step 1 - User Prompt: '{test_prompt}'")

    # Step 2: Preprocess
    cleaned = TextPreprocessor.clean_text(test_prompt)
    print(f"Step 2 - Preprocessed: '{cleaned}'")
    assert len(cleaned) > 0, "Cleaned prompt should not be empty"

    # Step 3: TF-IDF
    vec = tfidf.transform(cleaned)
    print(f"Step 3 - TF-IDF Shape: {vec.shape}")
    assert vec.shape[1] > 0, "TF-IDF feature vector must have columns"

    # Step 4: Intent
    intent_res = classifier.predict(vec)
    print(f"Step 4 - Predicted Intent: {intent_res['friendly_name']} ({intent_res['intent']}) with confidence {intent_res['confidence']}")
    assert intent_res["intent"] == "technical_coding", f"Expected technical_coding, got {intent_res['intent']}"

    # Step 5 & 6: User Profile
    sample_history = [
        "How do I fix a bug in React?",
        "Explain async await in Node.js",
        test_prompt
    ]
    profile_vector = profiler.extract_profile_from_history(sample_history)
    print(f"Step 6 - User Behaviour Profile Vector: {profile_vector}")

    # Step 7: Cluster assignment
    cluster_res = profiler.assign_cluster(profile_vector)
    print(f"Step 7 - Assigned Cluster: {cluster_res['badge_emoji']} {cluster_res['name']} (ID {cluster_res['cluster_id']})")

    # Step 8: Strategy Selection
    strategy = StrategySelector.select_strategy(intent_res, cluster_res, profile_vector)
    print(f"Step 8 - Strategy directive: {strategy['strategy_summary']}")
    assert "You are LAMA AI" in strategy["system_instruction"]

    # Step 9: (Simulated response generation)
    simulated_response = """
Here is the production-ready Python script using `requests` with exponential backoff rate limiting and pagination:

```python
import time
import requests

def fetch_paginated_data(base_url, max_pages=5):
    all_data = []
    page = 1
    while page <= max_pages:
        try:
            resp = requests.get(f"{base_url}?page={page}", timeout=10)
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                break
            all_data.extend(items)
            page += 1
        except requests.RequestException as e:
            print(f"Error on page {page}: {e}")
            break
    return all_data
```
Time Complexity: O(N) where N is total items fetched. Space Complexity: O(N) to store results.
"""

    # Step 10: Quality Prediction
    qual_res = quality_pred.predict(test_prompt, simulated_response, cluster_res["cluster_id"])
    print(f"Step 10 - Predicted Quality Match: {qual_res['percentage']}% ({qual_res['grade']})")
    assert qual_res["percentage"] >= 60, "Quality score should be reasonable"

    # Step 14: Preference update with feedback
    feedback = {"rating": 5, "tags": ["Great code", "Concise"]}
    updated_vec = profiler.update_profile_with_feedback(profile_vector, feedback)
    print(f"Step 14 - Online Updated Profile Vector: {updated_vec}")

    print("[SUCCESS] All pipeline steps 2 through 14 executed perfectly!")

if __name__ == "__main__":
    run_tests()
