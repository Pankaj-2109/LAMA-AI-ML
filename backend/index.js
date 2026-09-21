import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import ImageKit from "imagekit";
import mongoose from "mongoose";

import Chat from "./models/chat.js";
import UserChats from "./models/userChats.js";
import MLInteraction from "./models/mlInteraction.js";
import UserProfile from "./models/userProfile.js";

import { clerkMiddleware, getAuth } from "@clerk/express";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:5001";

const port = process.env.PORT || 3000;
const app = express();

// =======================
// CORS (MUST BE FIRST)
// =======================
const allowedOrigins = [
  "http://localhost:5173",
  "http://localhost:5174",
  "http://localhost:5175",
  "https://lama-ai-client1.vercel.app",
  process.env.CLIENT_URL,
].filter(Boolean);

app.use(
  cors({
    origin: function (origin, callback) {
      if (!origin) return callback(null, true);

      // Allow if it is in whitelisted list or ends with .vercel.app (handles Vercel branch preview URLs)
      const isAllowed = allowedOrigins.includes(origin) || origin.endsWith(".vercel.app") || origin.includes(".vercel.app");

      if (isAllowed) {
        return callback(null, true);
      }

      console.log("Blocked by CORS:", origin);
      return callback(null, false); // Reject origin gracefully without throwing Express middleware error
    },
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

app.options("*", cors());

// DEBUG
console.log(
  "CLERK_SECRET_KEY:",
  process.env.CLERK_SECRET_KEY
    ? process.env.CLERK_SECRET_KEY.slice(0, 10) + "..."
    : "MISSING"
);

// IMPORTANT
app.use(clerkMiddleware());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

//
// =======================
// BODY PARSER
// =======================
//

app.use(express.json());

//
// =======================
// MONGODB
// =======================
//

let isConnected = false;

const connect = async () => {
  if (isConnected && mongoose.connection.readyState === 1) {
    return;
  }
  try {
    console.log("MONGO =", process.env.MONGO ? "Defined" : "MISSING");
    if (!process.env.MONGO) {
      throw new Error("MONGO environment variable is not defined");
    }
    await mongoose.connect(process.env.MONGO);
    isConnected = true;
    console.log("Connected to MongoDB");
  } catch (err) {
    console.log("Mongo Error:", err);
    throw err;
  }
};

// Database connection middleware
app.use(async (req, res, next) => {
  // Skip DB connection for static files if any, but run for APIs
  if (req.path.startsWith("/api")) {
    try {
      await connect();
    } catch (err) {
      return res.status(500).send("Database connection error: " + err.message);
    }
  }
  next();
});

//
// =======================
// IMAGEKIT
// =======================
//

const imagekit = new ImageKit({
  urlEndpoint: process.env.IMAGE_KIT_ENDPOINT,
  publicKey: process.env.IMAGE_KIT_PUBLIC_KEY,
  privateKey: process.env.IMAGE_KIT_PRIVATE_KEY,
});

app.get("/api/upload", (req, res) => {
  const result = imagekit.getAuthenticationParameters();
  res.send(result);
});

//
// =======================
// CREATE CHAT
// =======================
//

app.post("/api/chats", async (req, res) => {
  try {
    const { userId } = getAuth(req);

    console.log("USER ID:", userId);
    console.log("BODY:", req.body);

    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    const { text, img } = req.body;

    const newChat = new Chat({
      userId,
      history: [
        {
          role: "user",
          parts: [{ text }],
          ...(img && { img }),
        },
      ],
    });

    const savedChat = await newChat.save();

    await UserChats.updateOne(
      { userId },
      {
        $push: {
          chats: {
            _id: savedChat._id,
            title: text.substring(0, 40),
          },
        },
      },
      { upsert: true }
    );

    res.status(201).send(savedChat._id);
  } catch (err) {
    console.log("CHAT ERROR:", err);
    res.status(500).send("Error creating chat: " + err.message);
  }
});

//
// =======================
// GET USER CHATS
// =======================
//

app.get("/api/userchats", async (req, res) => {
  try {
    const { userId } = getAuth(req);

    console.log("USER ID:", userId);

    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    const userChats = await UserChats.findOne({ userId });

    res.status(200).send(userChats?.chats || []);
  } catch (err) {
    console.log("USERCHATS ERROR:", err);
    res.status(500).send("Error fetching userchats: " + err.message);
  }
});

//
// =======================
// GET SINGLE CHAT
// =======================
//

app.get("/api/chats/:id", async (req, res) => {
  try {
    const { userId } = getAuth(req);

    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    const chat = await Chat.findOne({
      _id: req.params.id,
      userId,
    });

    res.status(200).send(chat);
  } catch (err) {
    console.log("GET CHAT ERROR:", err);
    res.status(500).send("Error fetching chat: " + err.message);
  }
});

//
// =======================
// UPDATE CHAT
// =======================
//

app.put("/api/chats/:id", async (req, res) => {
  try {
    const { userId } = getAuth(req);

    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    const { question, answer, img, mlMeta, feedback } = req.body;

    const modelPart = {
      role: "model",
      parts: [{ text: answer }],
      ...(mlMeta && { mlMeta }),
      ...(feedback && { feedback }),
    };

    const newItems = [
      ...(question !== null && question !== undefined
        ? [
          {
            role: "user",
            parts: [{ text: question }],
            ...(img && { img }),
          },
        ]
        : []),
      modelPart,
    ];

    const updatedChat = await Chat.findOneAndUpdate(
      { _id: req.params.id, userId },
      {
        $push: {
          history: {
            $each: newItems,
          },
        },
      },
      { new: true }
    );

    res.status(200).send(updatedChat);
  } catch (err) {
    console.log("UPDATE CHAT ERROR:", err);
    res.status(500).send("Error adding conversation: " + err.message);
  }
});

//
// =======================
// MACHINE LEARNING PIPELINE ENDPOINTS
// =======================
//

// Step 1 to 8: Pre-generation pipeline (Intent Classification, Profiling, K-Means Clustering, Strategy Selection)
app.post("/api/ml/pipeline", async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    const { prompt, chatId } = req.body;
    if (!prompt) {
      return res.status(400).send("Prompt is required");
    }

    // Step 5: Load the User's Chat History & User Profile from MongoDB
    let userHistory = [];
    try {
      const recentChats = await Chat.find({ userId }).sort({ updatedAt: -1 }).limit(3);
      for (const c of recentChats) {
        if (c.history) {
          for (const msg of c.history) {
            if (msg.role === "user" && msg.parts?.[0]?.text) {
              userHistory.push(msg.parts[0].text);
            }
          }
        }
      }
    } catch (dbErr) {
      console.warn("Could not fetch user history:", dbErr.message);
    }

    let userProfile = null;
    try {
      userProfile = await UserProfile.findOne({ userId });
    } catch (profileErr) {
      console.warn("Could not fetch user profile:", profileErr.message);
    }

    // Call Python ML service (Steps 2, 3, 4, 6, 7, 8)
    try {
      const mlResponse = await fetch(`${ML_SERVICE_URL}/ml/pipeline/pre-generation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          history: userHistory,
          user_profile_vector: userProfile?.vector || null,
        }),
      });

      if (mlResponse.ok) {
        const mlData = await mlResponse.json();

        // Update or create user profile with assigned cluster
        if (mlData.cluster && mlData.user_profile) {
          await UserProfile.findOneAndUpdate(
            { userId },
            {
              vector: mlData.user_profile.vector,
              clusterId: mlData.cluster.cluster_id,
              clusterName: mlData.cluster.name,
              badgeEmoji: mlData.cluster.badge_emoji,
              $inc: { totalInteractions: 1 },
              lastUpdated: new Date(),
            },
            { upsert: true, new: true }
          );
        }

        return res.status(200).json(mlData);
      }
    } catch (mlErr) {
      console.warn("ML Service unavailable, using default heuristic strategy:", mlErr.message);
    }

    // Graceful fallback if ML service is offline
    res.status(200).json({
      status: "fallback",
      intent: {
        intent: "conversational_casual",
        friendly_name: "General Conversational",
        confidence: 0.85,
      },
      cluster: {
        cluster_id: 0,
        name: "The Pragmatist",
        badge_emoji: "🎯",
        description: "Direct, helpful and structured.",
      },
      strategy: {
        strategy_summary: "Provide clear, structured, and helpful responses.",
        system_instruction: "You are LAMA AI. Respond with clear, direct, and helpful explanations.",
      },
    });
  } catch (err) {
    console.error("ML PIPELINE ERROR:", err);
    res.status(500).send("ML pipeline error: " + err.message);
  }
});

// Step 10 & 13: Predict Response Quality & Store Interaction
app.post("/api/ml/quality", async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    const { prompt, response, clusterId, chatId, mlMeta } = req.body;

    let prediction = {
      quality_score: 0.90,
      percentage: 90,
      grade: "Good",
    };

    try {
      const mlResponse = await fetch(`${ML_SERVICE_URL}/ml/predict-quality`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          response,
          cluster_id: clusterId || 0,
        }),
      });

      if (mlResponse.ok) {
        const mlData = await mlResponse.json();
        prediction = mlData.prediction;
      }
    } catch (mlErr) {
      console.warn("ML Quality service unavailable:", mlErr.message);
    }

    // Step 13: Store the Prompt, Prediction & metadata in MongoDB
    try {
      const newInteraction = new MLInteraction({
        userId,
        chatId: chatId || null,
        prompt,
        intent: mlMeta?.intent ? {
          name: mlMeta.intent,
          friendlyName: mlMeta.intentFriendly,
          confidence: mlMeta.intentConfidence,
        } : null,
        userCluster: {
          clusterId: mlMeta?.clusterId || 0,
          name: mlMeta?.clusterName || "The Pragmatist",
          badgeEmoji: mlMeta?.clusterEmoji || "🎯",
        },
        response,
        predictedQuality: {
          qualityScore: prediction.quality_score,
          percentage: prediction.percentage,
          grade: prediction.grade,
          features: prediction.features || {},
        },
      });

      await newInteraction.save();
    } catch (dbErr) {
      console.warn("Could not save ML interaction:", dbErr.message);
    }

    res.status(200).json({ status: "success", prediction });
  } catch (err) {
    console.error("ML QUALITY ERROR:", err);
    res.status(500).send("Quality prediction error: " + err.message);
  }
});

// Step 12, 13, 14: Collect User Feedback, Store It, and Update Preferences Online
app.post("/api/ml/feedback", async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    const { chatId, messageIndex, prompt, rating, thumbs, tags, comment } = req.body;

    const feedbackObj = {
      rating: rating || 5,
      thumbs: thumbs || null,
      tags: tags || [],
      comment: comment || "",
      createdAt: new Date(),
    };

    // Step 13: Store feedback in MLInteraction
    try {
      await MLInteraction.findOneAndUpdate(
        { userId, ...(prompt ? { prompt: prompt.trim() } : {}) },
        { $set: { feedback: feedbackObj } },
        { sort: { createdAt: -1 } }
      );
    } catch (saveErr) {
      console.warn("Could not update MLInteraction feedback:", saveErr.message);
    }

    // Update feedback inside Chat document
    if (chatId && messageIndex !== undefined) {
      try {
        await Chat.updateOne(
          { _id: chatId, userId },
          { $set: { [`history.${messageIndex}.feedback`]: feedbackObj } }
        );
      } catch (chatErr) {
        console.warn("Could not update Chat message feedback:", chatErr.message);
      }
    }

    // Step 14: Update User Preferences in Python ML service & MongoDB
    let updatedProfile = null;
    try {
      const userProfile = await UserProfile.findOne({ userId });
      const currentVector = userProfile?.vector || [40.0, 0.25, 0.4, 0.6, 4.0, 0.5];

      const updateResponse = await fetch(`${ML_SERVICE_URL}/ml/update-profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_vector: currentVector,
          feedback: feedbackObj,
        }),
      });

      if (updateResponse.ok) {
        const updateData = await updateResponse.json();
        updatedProfile = await UserProfile.findOneAndUpdate(
          { userId },
          {
            vector: updateData.updated_vector,
            clusterId: updateData.new_cluster.cluster_id,
            clusterName: updateData.new_cluster.name,
            badgeEmoji: updateData.new_cluster.badge_emoji,
            $inc: { feedbackCount: 1 },
            lastUpdated: new Date(),
          },
          { upsert: true, new: true }
        );
      }
    } catch (prefErr) {
      console.warn("Could not update online user preferences:", prefErr.message);
    }

    res.status(200).json({
      status: "success",
      message: "Feedback recorded and preferences adapted",
      profile: updatedProfile,
    });
  } catch (err) {
    console.error("ML FEEDBACK ERROR:", err);
    res.status(500).send("Feedback submission error: " + err.message);
  }
});

// Step 15: Retrieve Model Evaluation Metrics and System Analytics
app.get("/api/ml/metrics", async (req, res) => {
  try {
    let mlMetrics = {};
    try {
      const metricResp = await fetch(`${ML_SERVICE_URL}/ml/metrics`);
      if (metricResp.ok) {
        mlMetrics = await metricResp.json();
      }
    } catch (metricErr) {
      console.warn("Could not fetch metrics from Python ML service:", metricErr.message);
    }

    // Also get database stats
    let totalInteractions = 0;
    let avgFeedback = 4.5;
    let recentInteractions = [];

    try {
      totalInteractions = await MLInteraction.countDocuments();
      recentInteractions = await MLInteraction.find().sort({ createdAt: -1 }).limit(10);
    } catch (e) {
      // ignore
    }

    res.status(200).json({
      status: "success",
      metrics: mlMetrics,
      database_stats: {
        totalInteractions,
        avgFeedback,
        recentInteractions,
      },
    });
  } catch (err) {
    console.error("ML METRICS ERROR:", err);
    res.status(500).send("Metrics fetch error: " + err.message);
  }
});

// Step 15: Retrain ML Models
app.post("/api/ml/retrain", async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    // Collect extra logged prompts from MongoDB to include in retraining
    let extraIntents = [];
    try {
      const interactions = await MLInteraction.find({ "intent.name": { $exists: true } }).limit(200);
      extraIntents = interactions
        .filter((i) => i.prompt && i.intent?.name)
        .map((i) => [i.prompt, i.intent.name]);
    } catch (e) {
      console.warn("Could not collect extra intents from DB:", e.message);
    }

    const retrainResp = await fetch(`${ML_SERVICE_URL}/ml/retrain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        extra_intents: extraIntents.length > 0 ? extraIntents : null,
      }),
    });

    if (!retrainResp.ok) {
      throw new Error("Failed to trigger ML retraining");
    }

    const retrainData = await retrainResp.json();
    res.status(200).json(retrainData);
  } catch (err) {
    console.error("ML RETRAIN ERROR:", err);
    res.status(500).send("Model retraining error: " + err.message);
  }
});

// Get User Profile & Persona
app.get("/api/ml/profile", async (req, res) => {
  try {
    const { userId } = getAuth(req);
    if (!userId) {
      return res.status(401).send("Unauthorized");
    }

    let profile = await UserProfile.findOne({ userId });
    if (!profile) {
      profile = await UserProfile.create({
        userId,
        vector: [40.0, 0.25, 0.4, 0.6, 4.0, 0.5],
        clusterId: 0,
        clusterName: "The Pragmatist",
        badgeEmoji: "🎯",
      });
    }

    res.status(200).json({ status: "success", profile });
  } catch (err) {
    console.error("ML PROFILE ERROR:", err);
    res.status(500).send("Profile error: " + err.message);
  }
});

//
// =======================
// ERROR HANDLER
// =======================
//

app.use((err, req, res, next) => {
  console.error("GLOBAL ERROR:", err);
  res.status(401).send("Unauthenticated!");
});

//
// =======================
// FRONTEND BUILD
// =======================
//

// Only serve static files in non-serverless environments (local dev / traditional hosting)
if (!process.env.VERCEL) {
  app.use(express.static(path.join(__dirname, "../client/dist")));
  app.get("*", (req, res) => {
    res.sendFile(path.join(__dirname, "../client/dist", "index.html"));
  });
}

//
// =======================
// START SERVER
// =======================
//

app.listen(port, () => {
  connect();
  console.log(`Server running on ${port}`);
});
// Nodemon reload trigger