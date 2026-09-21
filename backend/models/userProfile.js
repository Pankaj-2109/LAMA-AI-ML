import mongoose from "mongoose";

const userProfileSchema = new mongoose.Schema(
  {
    userId: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    vector: {
      type: [Number],
      default: [40.0, 0.25, 0.4, 0.6, 4.0, 0.5],
    },
    clusterId: {
      type: Number,
      default: 0,
    },
    clusterName: {
      type: String,
      default: "The Pragmatist",
    },
    badgeEmoji: {
      type: String,
      default: "🎯",
    },
    totalInteractions: {
      type: Number,
      default: 0,
    },
    averageRating: {
      type: Number,
      default: 4.0,
    },
    feedbackCount: {
      type: Number,
      default: 0,
    },
    lastUpdated: {
      type: Date,
      default: Date.now,
    },
  },
  { timestamps: true }
);

export default mongoose.models.userprofile ||
  mongoose.model("userprofile", userProfileSchema);
