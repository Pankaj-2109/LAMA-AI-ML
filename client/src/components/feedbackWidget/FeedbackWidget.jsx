import { useState } from "react";
import "./feedbackWidget.css";
import { useAuth } from "@clerk/clerk-react";

const TAG_OPTIONS = [
  "🎯 Concise",
  "⚡ Detailed",
  "✅ Accurate",
  "💡 Great Ideas",
  "❌ Too Long",
  "⚠️ Needs Detail",
];

const FeedbackWidget = ({ chatId, messageIndex, prompt, initialFeedback }) => {
  const { getToken } = useAuth();
  const [rating, setRating] = useState(initialFeedback?.rating || 0);
  const [hoverRating, setHoverRating] = useState(0);
  const [thumbs, setThumbs] = useState(initialFeedback?.thumbs || null);
  const [selectedTags, setSelectedTags] = useState(initialFeedback?.tags || []);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(!!initialFeedback?.rating);

  const toggleTag = (tag) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const submitFeedback = async (explicitRating = rating, explicitThumbs = thumbs) => {
    setIsSubmitting(true);
    try {
      const token = await getToken();
      await fetch(
        `${import.meta.env.VITE_API_URL || "http://localhost:3000"}/api/ml/feedback`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            chatId,
            messageIndex,
            prompt,
            rating: explicitRating || 5,
            thumbs: explicitThumbs,
            tags: selectedTags,
          }),
        }
      );
      setIsSubmitted(true);
    } catch (err) {
      console.error("Failed to submit feedback:", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRatingClick = (val) => {
    setRating(val);
    submitFeedback(val, thumbs);
  };

  const handleThumbsClick = (val) => {
    const newThumbs = thumbs === val ? null : val;
    setThumbs(newThumbs);
    submitFeedback(rating, newThumbs);
  };

  return (
    <div className="feedbackWidget">
      <div className="feedbackHeader">
        <span className="feedbackPrompt">Rate this response:</span>

        {/* Thumbs up / down */}
        <div className="thumbsGroup">
          <button
            type="button"
            className={`thumbBtn ${thumbs === "up" ? "active" : ""}`}
            onClick={() => handleThumbsClick("up")}
            title="Helpful"
          >
            👍
          </button>
          <button
            type="button"
            className={`thumbBtn ${thumbs === "down" ? "active" : ""}`}
            onClick={() => handleThumbsClick("down")}
            title="Unhelpful"
          >
            👎
          </button>
        </div>

        {/* 5-Star Rating */}
        <div className="starsGroup">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              type="button"
              className={`starBtn ${(hoverRating || rating) >= star ? "filled" : ""}`}
              onMouseEnter={() => setHoverRating(star)}
              onMouseLeave={() => setHoverRating(0)}
              onClick={() => handleRatingClick(star)}
              title={`${star} star${star > 1 ? "s" : ""}`}
            >
              ★
            </button>
          ))}
        </div>

        {isSubmitted && (
          <span className="feedbackSuccess">✓ Preferences updated</span>
        )}
      </div>

      {/* Quick feedback tags */}
      <div className="tagsGroup">
        {TAG_OPTIONS.map((tag) => (
          <button
            key={tag}
            type="button"
            className={`tagPill ${selectedTags.includes(tag) ? "selected" : ""}`}
            onClick={() => {
              toggleTag(tag);
              // Submit after toggle
              setTimeout(() => submitFeedback(rating, thumbs), 100);
            }}
          >
            {tag}
          </button>
        ))}
      </div>
    </div>
  );
};

export default FeedbackWidget;
