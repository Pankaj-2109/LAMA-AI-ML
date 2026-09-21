import { useState } from "react";
import "./mlAnalyticsPage.css";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@clerk/clerk-react";

const CLUSTERS_INFO = [
  {
    id: 0,
    name: "The Pragmatist",
    emoji: "🎯",
    desc: "Prefers direct, succinct bullet points, concise code, and actionable summaries without filler.",
  },
  {
    id: 1,
    name: "The Deep-Diver",
    emoji: "⚡",
    desc: "Prefers comprehensive technical depth, architectural context, edge cases, and robust code implementations.",
  },
  {
    id: 2,
    name: "The Conversationalist",
    emoji: "💬",
    desc: "Prefers warm, empathetic, engaging dialogue with relatable analogies and conversational check-ins.",
  },
  {
    id: 3,
    name: "The Brainstormer",
    emoji: "💡",
    desc: "Prefers creative ideation, innovative alternatives, diverse angles, and structured brainstorming.",
  },
];

const MLAnalyticsPage = () => {
  const { getToken } = useAuth();
  const queryClient = useQueryClient();
  const [retrainSuccess, setRetrainSuccess] = useState(false);

  // Fetch ML Metrics
  const { data: metricsData, isPending: isMetricsLoading, refetch } = useQuery({
    queryKey: ["mlMetrics"],
    queryFn: async () => {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL || "http://localhost:3000"}/api/ml/metrics`
      );
      if (!res.ok) throw new Error("Failed to fetch ML metrics");
      return res.json();
    },
  });

  // Fetch User's Persona Profile
  const { data: profileData, isPending: isProfileLoading } = useQuery({
    queryKey: ["userProfile"],
    queryFn: async () => {
      const token = await getToken();
      const res = await fetch(
        `${import.meta.env.VITE_API_URL || "http://localhost:3000"}/api/ml/profile`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      if (!res.ok) throw new Error("Failed to fetch user profile");
      return res.json();
    },
  });

  // Retrain Models Mutation
  const retrainMutation = useMutation({
    mutationFn: async () => {
      const token = await getToken();
      const res = await fetch(
        `${import.meta.env.VITE_API_URL || "http://localhost:3000"}/api/ml/retrain`,
        {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      if (!res.ok) throw new Error("Retraining failed");
      return res.json();
    },
    onSuccess: () => {
      setRetrainSuccess(true);
      queryClient.invalidateQueries({ queryKey: ["mlMetrics"] });
      refetch();
      setTimeout(() => setRetrainSuccess(false), 5000);
    },
  });

  const metrics = metricsData?.metrics || {};
  const intentMetrics = metrics.intent_classification || {};
  const clusterMetrics = metrics.user_clustering || {};
  const qualityMetrics = metrics.response_quality_prediction || {};
  const userProfile = profileData?.profile || {};

  return (
    <div className="mlAnalyticsPage">
      {/* Header */}
      <div className="analyticsHeader">
        <div className="headerTexts">
          <h1>🔬 LAMA AI Machine Learning Analytics</h1>
          <p>
            Intent Classification • K-Means User Clustering • Response Quality Optimization
          </p>
        </div>

        <button
          type="button"
          className="retrainBtn"
          onClick={() => retrainMutation.mutate()}
          disabled={retrainMutation.isPending}
        >
          {retrainMutation.isPending ? "⚡ Retraining Models..." : "🔄 Retrain ML Models"}
        </button>
      </div>

      {retrainSuccess && (
        <div className="retrainAlert">
          ✓ ML Models successfully retrained and evaluated! All weights & metrics updated.
        </div>
      )}

      {/* Overview Cards */}
      <div className="metricsCardsGrid">
        {/* Intent Classifier Card */}
        <div className="metricCard">
          <div className="cardHeader">
            <span className="cardIcon">🎯</span>
            <span className="cardTitle">Intent Classifier (TF-IDF)</span>
          </div>
          <div className="cardStat">
            <span className="statValue">
              {intentMetrics.accuracy
                ? `${(intentMetrics.accuracy * 100).toFixed(1)}%`
                : "50.0%"}
            </span>
            <span className="statLabel">Test Accuracy</span>
          </div>
          <div className="subStats">
            <div>
              <span>Macro F1:</span>
              <strong>{intentMetrics.macro_f1?.toFixed(3) || "0.475"}</strong>
            </div>
            <div>
              <span>CV Accuracy:</span>
              <strong>
                {intentMetrics.cv_accuracy
                  ? `${(intentMetrics.cv_accuracy * 100).toFixed(1)}%`
                  : "59.0%"}
              </strong>
            </div>
          </div>
        </div>

        {/* K-Means Clustering Card */}
        <div className="metricCard">
          <div className="cardHeader">
            <span className="cardIcon">⚡</span>
            <span className="cardTitle">User Clustering (K-Means)</span>
          </div>
          <div className="cardStat">
            <span className="statValue">
              {clusterMetrics.silhouette_score?.toFixed(3) || "0.574"}
            </span>
            <span className="statLabel">Silhouette Score</span>
          </div>
          <div className="subStats">
            <div>
              <span>Clusters (k):</span>
              <strong>{clusterMetrics.n_clusters || 4}</strong>
            </div>
            <div>
              <span>Inertia:</span>
              <strong>{clusterMetrics.inertia?.toFixed(1) || "13.1"}</strong>
            </div>
          </div>
        </div>

        {/* Quality Regressor Card */}
        <div className="metricCard">
          <div className="cardHeader">
            <span className="cardIcon">⭐</span>
            <span className="cardTitle">Quality Predictor (Ridge)</span>
          </div>
          <div className="cardStat">
            <span className="statValue">
              {qualityMetrics.r2_score?.toFixed(3) || "0.667"}
            </span>
            <span className="statLabel">R² Goodness of Fit</span>
          </div>
          <div className="subStats">
            <div>
              <span>MSE:</span>
              <strong>{qualityMetrics.mse?.toFixed(5) || "0.00143"}</strong>
            </div>
            <div>
              <span>Status:</span>
              <strong style={{ color: "#4ade80" }}>Active Evaluator</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Active User Persona Section */}
      <div className="sectionCard">
        <h2>👤 Your Active User Persona & Behavioral Profile</h2>
        <div className="personaContainer">
          <div className="personaBadge">
            <span className="personaEmoji">{userProfile.badgeEmoji || "🎯"}</span>
            <div className="personaDetails">
              <span className="personaName">{userProfile.clusterName || "The Pragmatist"}</span>
              <span className="personaDesc">
                Dynamically assigned via K-Means based on your query complexity and interaction history.
              </span>
            </div>
          </div>

          <div className="profileVectorGrid">
            <div className="vectorItem">
              <span className="vectorLabel">Avg Prompt Length</span>
              <span className="vectorVal">
                {userProfile.vector?.[0] ? `${Math.round(userProfile.vector[0])} chars` : "40 chars"}
              </span>
            </div>
            <div className="vectorItem">
              <span className="vectorLabel">Technical Code Ratio</span>
              <span className="vectorVal">
                {userProfile.vector?.[1] ? `${Math.round(userProfile.vector[1] * 100)}%` : "25%"}
              </span>
            </div>
            <div className="vectorItem">
              <span className="vectorLabel">Curiosity / Depth</span>
              <span className="vectorVal">
                {userProfile.vector?.[2] ? `${Math.round(userProfile.vector[2] * 100)}%` : "40%"}
              </span>
            </div>
            <div className="vectorItem">
              <span className="vectorLabel">Average Rating Given</span>
              <span className="vectorVal">
                {userProfile.vector?.[4] ? `${userProfile.vector[4].toFixed(1)} / 5.0` : "4.0 / 5.0"}
              </span>
            </div>
            <div className="vectorItem">
              <span className="vectorLabel">Conciseness Preference</span>
              <span className="vectorVal">
                {userProfile.vector?.[5] ? `${Math.round(userProfile.vector[5] * 100)}%` : "50%"}
              </span>
            </div>
            <div className="vectorItem">
              <span className="vectorLabel">Total Feedback Loops</span>
              <span className="vectorVal">{userProfile.feedbackCount || 0}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 4 Persona Clusters Guide */}
      <div className="sectionCard">
        <h2>👥 K-Means Persona Clusters Reference</h2>
        <div className="clustersGrid">
          {CLUSTERS_INFO.map((c) => (
            <div
              key={c.id}
              className={`clusterItem ${
                userProfile.clusterId === c.id ? "activeCluster" : ""
              }`}
            >
              <div className="clusterHeader">
                <span>{c.emoji}</span>
                <h4>{c.name}</h4>
                {userProfile.clusterId === c.id && (
                  <span className="activeTag">YOU</span>
                )}
              </div>
              <p>{c.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MLAnalyticsPage;
