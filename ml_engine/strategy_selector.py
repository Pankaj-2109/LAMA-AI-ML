"""
Step 8: Personalized Response Strategy Selector for LAMA AI.
Combines Prompt Intent, User Persona Cluster, and Preference Vector into targeted Gemini System Instructions.
"""

INTENT_STRATEGY_MAP = {
    "technical_coding": {
        "focus": "Focus on high-quality code, syntax precision, and optimal complexity.",
        "pragmatist": "Deliver clean, production-ready code with minimal explanation. State time and space complexity in 1 line.",
        "deep_diver": "Provide clean code along with architectural design patterns, edge-case analysis, and underlying runtime mechanics.",
        "conversationalist": "Provide well-commented code explained with intuitive, step-by-step guidance.",
        "brainstormer": "Provide multiple implementation paradigms (e.g. functional vs OOP, iterative vs recursive) with trade-offs."
    },
    "conceptual_educational": {
        "focus": "Focus on clarity, foundational principles, and educational value.",
        "pragmatist": "Explain the concept in 3-5 succinct bullet points with a TL;DR summary.",
        "deep_diver": "Provide an in-depth breakdown covering foundational theory, historical context, diagrams/tables, and advanced nuances.",
        "conversationalist": "Explain using friendly everyday analogies, relatable examples, and check-in questions.",
        "brainstormer": "Explore the concept across different disciplines, future applications, and provocative thought experiments."
    },
    "conversational_casual": {
        "focus": "Focus on natural interaction, personality, and active listening.",
        "pragmatist": "Be polite, direct, and brief.",
        "deep_diver": "Engage thoughtfully with intellectual curiosity and interesting trivia.",
        "conversationalist": "Be warm, empathetic, cheerful, and conversational. Ask an engaging follow-up question.",
        "brainstormer": "Respond with humor, clever wit, and playful ideas."
    },
    "problem_solving_advice": {
        "focus": "Focus on actionable steps, decision frameworks, and troubleshooting.",
        "pragmatist": "Give an immediate numbered action checklist: Step 1, Step 2, Step 3.",
        "deep_diver": "Provide root-cause analysis, systematic troubleshooting tree, and long-term preventative measures.",
        "conversationalist": "Provide reassuring, empathetic guidance with practical suggestions.",
        "brainstormer": "Offer conventional and unconventional solutions, evaluating pros and cons for each."
    },
    "creative_brainstorming": {
        "focus": "Focus on novelty, rich vocabulary, and diverse perspectives.",
        "pragmatist": "Present a curated list of top high-impact ideas with one-sentence rationale each.",
        "deep_diver": "Present structured concepts with full world-building, thematic elements, or market viability.",
        "conversationalist": "Bounce creative ideas back and forth in an encouraging brainstorming session.",
        "brainstormer": "Generate bold, unconventional, and diverse ideas across completely different angles."
    },
    "factual_inquiry": {
        "focus": "Focus on factual accuracy, verified sources, and precision.",
        "pragmatist": "Give the exact answer immediately in the first sentence.",
        "deep_diver": "Provide the factual answer plus background context, timeline, and verified specifics.",
        "conversationalist": "Provide the fact warmly with an interesting related piece of trivia.",
        "brainstormer": "Provide the fact and highlight intriguing questions or paradoxes related to it."
    }
}

class StrategySelector:
    """Step 8: Generates adaptive system instructions for Gemini generation."""

    @staticmethod
    def select_strategy(intent_data: dict, cluster_data: dict, profile_vector: list = None) -> dict:
        intent = intent_data.get("intent", "conversational_casual")
        cluster_id = cluster_data.get("cluster_id", 0)
        cluster_name = cluster_data.get("name", "The Pragmatist")

        # Map cluster ID to key
        cluster_keys = {
            0: "pragmatist",
            1: "deep_diver",
            2: "conversationalist",
            3: "brainstormer"
        }
        cluster_key = cluster_keys.get(cluster_id, "pragmatist")

        intent_info = INTENT_STRATEGY_MAP.get(intent, INTENT_STRATEGY_MAP["conversational_casual"])
        strategy_directive = intent_info.get(cluster_key, cluster_data.get("style_directive", ""))

        system_instruction = (
            f"You are LAMA AI, an advanced personalized AI chatbot.\n"
            f"[User Persona Context]: The user belongs to persona '{cluster_name}'. {cluster_data.get('description', '')}\n"
            f"[Detected Intent]: {intent_data.get('friendly_name', intent)}.\n"
            f"[Adaptation Directive]: {strategy_directive}\n"
            f"Always follow this personalization directive while maintaining correctness and helpfulness."
        )

        return {
            "intent": intent,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "badge_emoji": cluster_data.get("badge_emoji", "🤖"),
            "strategy_summary": strategy_directive,
            "system_instruction": system_instruction
        }
