import mongoose from "mongoose";

const mlInteractionSchema = new mongoose.Schema(
  {
    userId: {
      type: String,
      required: true,
      index: true,
    },
    chatId: {
      type: String,
      required: false,
    },
    prompt: {
      type: String,
      required: true,
    },
    preprocessedPrompt: {
      type: String,
    },
    intent: {
      name: String,
      friendlyName: String,
      confidence: Number,
    },
    userCluster: {
      clusterId: Number,
      name: String,
      badgeEmoji: String,
    },
    strategy: {
      summary: String,
      systemInstruction: String,
    },
    response: {
      type: String,
      default: "",
    },
    predictedQuality: {
      qualityScore: Number,
      percentage: Number,
      grade: String,
      features: Object,
    },
    feedback: {
      rating: { type: Number, min: 1, max: 5 },
      thumbs: { type: String, enum: ["up", "down", null], default: null },
      tags: [String],
      comment: String,
      createdAt: { type: Date, default: Date.now },
    },
  },
  { timestamps: true }
);

export default mongoose.models.mlinteraction ||
  mongoose.model("mlinteraction", mlInteractionSchema);
