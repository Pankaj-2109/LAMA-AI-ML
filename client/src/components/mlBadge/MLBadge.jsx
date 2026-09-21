import { useState } from "react";
import "./mlBadge.css";

const MLBadge = ({ mlMeta }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!mlMeta || (!mlMeta.intent && !mlMeta.clusterName)) {
    return null;
  }

  const {
    intentFriendly,
    intent,
    intentConfidence,
    clusterName,
    clusterEmoji,
    predictedQuality,
    qualityGrade,
  } = mlMeta;

  const confidencePct = intentConfidence
    ? Math.round(intentConfidence * 100)
    : null;

  return (
    <div className="mlBadgeContainer">
      <div className="mlBadgeRow" onClick={() => setIsOpen(!isOpen)} role="button" tabIndex={0}>
        {/* Intent Badge */}
        {intent && (
          <span className="badge intentBadge" title="Predicted User Intent">
            <span className="badgeIcon">🎯</span>
            <span className="badgeLabel">{intentFriendly || intent}</span>
            {confidencePct && <span className="badgeSub">{confidencePct}%</span>}
          </span>
        )}

        {/* Persona Cluster Badge */}
        {clusterName && (
          <span className="badge clusterBadge" title="User Behaviour Persona Cluster (K-Means)">
            <span className="badgeIcon">{clusterEmoji || "⚡"}</span>
            <span className="badgeLabel">{clusterName}</span>
          </span>
        )}

        {/* Predicted Quality Badge */}
        {predictedQuality !== undefined && predictedQuality !== null && (
          <span className="badge qualityBadge" title="ML Predicted Response Quality">
            <span className="badgeIcon">⭐</span>
            <span className="badgeLabel">{predictedQuality}% Match</span>
            {qualityGrade && <span className="badgeSub">({qualityGrade})</span>}
          </span>
        )}

        <span className="toggleIndicator">{isOpen ? "▲ Details" : "▼ ML Telemetry"}</span>
      </div>

      {/* Expanded Telemetry Drawer */}
      {isOpen && (
        <div className="mlTelemetryDrawer">
          <div className="telemetryTitle">
            <span>🔬 Machine Learning Pipeline Telemetry</span>
          </div>

          <div className="telemetryGrid">
            <div className="telemetryItem">
              <span className="telemetryKey">Step 3 & 4: Intent Classification</span>
              <span className="telemetryValue">
                {intentFriendly || intent} ({confidencePct ? `${confidencePct}% confidence` : "N/A"}) via TF-IDF + Classifier
              </span>
            </div>

            <div className="telemetryItem">
              <span className="telemetryKey">Step 6 & 7: User Clustering</span>
              <span className="telemetryValue">
                Assigned to <strong>{clusterName}</strong> based on prompt patterns and interaction depth via K-Means (k=4).
              </span>
            </div>

            <div className="telemetryItem">
              <span className="telemetryKey">Step 8 & 9: Adaptive Strategy</span>
              <span className="telemetryValue">
                Personalized system directives injected dynamically into Gemini generative stream.
              </span>
            </div>

            <div className="telemetryItem">
              <span className="telemetryKey">Step 10: Response Quality Prediction</span>
              <span className="telemetryValue">
                {predictedQuality}% quality match evaluated via cross-feature regression model.
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MLBadge;
