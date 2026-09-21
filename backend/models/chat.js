import mongoose from "mongoose";

const chatSchema = new mongoose.Schema(
  {
    userId: {
      type: String,
      required: true,
    },
    history: [
      {
        role: {
          type: String,
          enum: ["user", "model"],
          required: true,
        },
        parts: [
          {
            text: {
              type: String,
              required: true,
            },
          },
        ],
        img: {
          type: String,
          required: false,
        },
        mlMeta: {
          intent: { type: String },
          intentFriendly: { type: String },
          intentConfidence: { type: Number },
          clusterId: { type: Number },
          clusterName: { type: String },
          clusterEmoji: { type: String },
          predictedQuality: { type: Number },
          qualityGrade: { type: String },
        },
        feedback: {
          rating: { type: Number },
          thumbs: { type: String },
          tags: [{ type: String }],
          createdAt: { type: Date },
        },
      },
    ],
  },
  { timestamps: true }
);

export default mongoose.models.chat || mongoose.model("chat", chatSchema);
