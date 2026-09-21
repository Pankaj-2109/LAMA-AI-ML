import { useEffect, useRef, useState } from "react";
import "./newPrompt.css";
import Upload from "../upload/Upload";
import { IKImage } from "imagekitio-react";
import { generateGeminiStream } from "../../lib/gemini";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@clerk/clerk-react";

const NewPrompt = ({ data, setMessages }) => {
  const { getToken } = useAuth();
  const [isThinking, setIsThinking] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const [img, setImg] = useState({
    isLoading: false,
    dbData: {},
    aiData: {},
  });

  const formRef = useRef(null);
  const queryClient = useQueryClient();

  // Save chat with ML metadata
  const mutation = useMutation({
    mutationFn: async ({ question, answer, mlMeta }) => {
      const token = await getToken();
      const res = await fetch(
        `${import.meta.env.VITE_API_URL || "http://localhost:3000"}/api/chats/${data._id}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            question,
            answer,
            img: img.dbData?.filePath || null,
            mlMeta,
          }),
        }
      );

      if (!res.ok) throw new Error("Failed to save chat");
      return res.json();
    },

    onSuccess: (newChatData) => {
      queryClient.setQueryData(["chat", data._id], newChatData);
      setMessages([]);
      setImg({ isLoading: false, dbData: {}, aiData: {} });
    },
  });

  const hasTriggeredRef = useRef(null);

  // Consolidated response generator incorporating LAMA AI ML Pipeline
  const generateResponse = async (text, isAuto = false) => {
    if ((!text?.trim() && !img.dbData?.filePath) || isThinking) return;

    setIsThinking(true);

    if (isAuto) {
      setMessages([{ role: "assistant", content: "", mlMeta: null }]);
    } else {
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content: text,
          img: img.dbData?.filePath || null,
        },
        { role: "assistant", content: "", mlMeta: null },
      ]);
    }

    try {
      const token = await getToken();

      // Step 1 - 8: Invoke ML Pipeline (Preprocessing, TF-IDF, Intent Classification, Profiling, K-Means, Strategy Selection)
      let pipelineData = null;
      try {
        const mlRes = await fetch(
          `${import.meta.env.VITE_API_URL || "http://localhost:3000"}/api/ml/pipeline`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              prompt: text || "Analyze image",
              chatId: data?._id,
            }),
          }
        );
        if (mlRes.ok) {
          pipelineData = await mlRes.json();
        }
      } catch (mlErr) {
        console.warn("Could not retrieve ML strategy:", mlErr);
      }

      const systemInstruction = pipelineData?.strategy?.system_instruction || null;

      const history = data?.history || [];
      const baseHistory = isAuto ? history.slice(0, -1) : history;

      const chatHistory = baseHistory.map(({ role, parts }) => ({
        role,
        parts: [{ text: parts?.[0]?.text || "" }],
      }));

      const messageParts = Object.keys(img.aiData || {}).length
        ? (text?.trim() ? [img.aiData, text] : [img.aiData])
        : [text];

      // Step 9: Generate response using Gemini conditioned on the personalized systemInstruction
      const result = await generateGeminiStream(chatHistory, messageParts, systemInstruction);

      let fullText = "";

      for await (const chunk of result.stream) {
        fullText += chunk.text();

        setMessages((prev) => [
          ...prev.slice(0, -1),
          {
            role: "assistant",
            content: fullText,
            mlMeta: pipelineData
              ? {
                  intent: pipelineData.intent?.intent,
                  intentFriendly: pipelineData.intent?.friendly_name,
                  intentConfidence: pipelineData.intent?.confidence,
                  clusterId: pipelineData.cluster?.cluster_id,
                  clusterName: pipelineData.cluster?.name,
                  clusterEmoji: pipelineData.cluster?.badge_emoji,
                }
              : null,
          },
        ]);
      }

      // Step 10: Predict Response Quality via ML cross-feature model
      let qualityData = null;
      try {
        const qualRes = await fetch(
          `${import.meta.env.VITE_API_URL || "http://localhost:3000"}/api/ml/quality`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              prompt: text || "Image query",
              response: fullText,
              clusterId: pipelineData?.cluster?.cluster_id || 0,
              chatId: data?._id,
              mlMeta: {
                intent: pipelineData?.intent?.intent,
                intentFriendly: pipelineData?.intent?.friendly_name,
                intentConfidence: pipelineData?.intent?.confidence,
                clusterId: pipelineData?.cluster?.cluster_id,
                clusterName: pipelineData?.cluster?.name,
                clusterEmoji: pipelineData?.cluster?.badge_emoji,
              },
            }),
          }
        );
        if (qualRes.ok) {
          qualityData = await qualRes.json();
        }
      } catch (qualErr) {
        console.warn("Could not predict quality:", qualErr);
      }

      const finalMlMeta = {
        intent: pipelineData?.intent?.intent || "conversational_casual",
        intentFriendly: pipelineData?.intent?.friendly_name || "Conversational & Casual",
        intentConfidence: pipelineData?.intent?.confidence || 0.85,
        clusterId: pipelineData?.cluster?.cluster_id || 0,
        clusterName: pipelineData?.cluster?.name || "The Pragmatist",
        clusterEmoji: pipelineData?.cluster?.badge_emoji || "🎯",
        predictedQuality: qualityData?.prediction?.percentage || 92,
        qualityGrade: qualityData?.prediction?.grade || "Good",
      };

      // Step 11: Display with updated ML telemetry
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: fullText, mlMeta: finalMlMeta },
      ]);

      // Step 13: Store the Turn & ML Metadata in MongoDB
      mutation.mutate({
        question: isAuto ? null : text,
        answer: fullText,
        mlMeta: finalMlMeta,
      });

    } catch (err) {
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: "❌ " + (err.message || "Error"), mlMeta: null },
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  // Auto-respond to the first message if no assistant response exists
  useEffect(() => {
    const history = data?.history || [];
    if (history.length > 0 && history[history.length - 1].role === "user") {
      const triggerKey = `${data._id}_${history.length}`;
      if (hasTriggeredRef.current !== triggerKey) {
        hasTriggeredRef.current = triggerKey;
        const lastMsg = history[history.length - 1];
        generateResponse(lastMsg.parts?.[0]?.text || "", true);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = inputValue;
    const hasImage = !!img.dbData?.filePath;
    if (!text.trim() && !hasImage) return;
    generateResponse(text);
    setInputValue("");
  };

  return (
    <form
      className="newForm"
      onSubmit={handleSubmit}
      ref={formRef}
      autoComplete="off"
    >
      {/* IMAGE PREVIEW IN THE CAPSULE */}
      {(img.isLoading || img.dbData?.filePath) && (
        <div className="imagePreviewArea">
          {img.isLoading ? (
            <div className="imagePreviewLoading">
              <span>Uploading...</span>
            </div>
          ) : (
            <div className="imagePreviewWrapper">
              <IKImage
                urlEndpoint={import.meta.env.VITE_IMAGE_KIT_ENDPOINT}
                path={img.dbData.filePath}
                width="60"
                height="60"
              />
              <button
                type="button"
                className="removeImageBtn"
                onClick={() => setImg({ isLoading: false, dbData: {}, aiData: {} })}
              >
                ✕
              </button>
            </div>
          )}
        </div>
      )}

      <div className="inputContainer">
        <Upload setImg={setImg} />

        <input
          type="text"
          name="chat_input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={isThinking ? "Thinking..." : "Ask me anything..."}
          autoComplete="off"
          disabled={isThinking}
        />

        <button
          type="submit"
          disabled={isThinking || (!inputValue.trim() && !img.dbData?.filePath)}
          className={(inputValue.trim() || img.dbData?.filePath) ? "glow" : ""}
        >
          <img src="/arrow.png" alt="Send" />
        </button>
      </div>
    </form>
  );
};

export default NewPrompt;