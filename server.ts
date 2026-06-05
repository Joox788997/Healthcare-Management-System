import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Lazy load Gemini AI to protect startup.
  let aiInstance: GoogleGenAI | null = null;
  function getAI() {
    if (!aiInstance) {
      const key = process.env.GEMINI_API_KEY;
      if (!key) {
        console.warn("GEMINI_API_KEY environment variable is not defined. Symptom Checker will run in mocked offline mode.");
        return null;
      }
      aiInstance = new GoogleGenAI({
        apiKey: key,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          }
        }
      });
    }
    return aiInstance;
  }

  // REST API endpoint for symptom analysis using Gemini 3.5 Flash
  app.post("/api/analyze-symptoms", async (req, res) => {
    try {
      const { symptoms } = req.body;
      if (!symptoms || typeof symptoms !== "string" || symptoms.trim().length === 0) {
        return res.status(400).json({ error: "Symptom description is required." });
      }

      const ai = getAI();
      if (!ai) {
        // Return a simulated, helpful responsive JSON structure in case GEMINI_API_KEY is not defined
        console.warn("Generating mock symptom checking response since GEMINI_API_KEY is missing");
        const mockResult = {
          conditions: [
            { name: "Mild Viral Syndrome / Common Cold", confidence: 85 },
            { name: "Allergic Rhinitis (Seasonal Allergies)", confidence: 60 },
            { name: "Acute Sinusitis", confidence: 45 }
          ],
          specialization: "General Physician / Family Doctor",
          urgency: "Low",
          details: "This is a simulated mock analysis because the environment is running offline. Keep hydrated, rest, and verify your actual symptoms in person if they worsen."
        };
        return res.json(mockResult);
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: `You are a medical triage assistant. Analyze these symptoms: "${symptoms}". Provide a structured, parseable JSON response (no markdown blocks, just the JSON string itself) with exactly the following keys:
- conditions: an array of objects, where each object has: "name" (string) and "confidence" (integer percentage from 1 to 100)
- specialization: recommended hospital medical department/specialist (e.g., Cardiology, Pulmonology, General Medicine)
- urgency: one of "Low", "Medium", and "High"
- details: brief clinical advice/summary in 2 sentences.

Do not declare any diagnosis, instead present likely causes with caution. Provide only the JSON format without any enclosing markdown backticks.`,
        config: {
          responseMimeType: "application/json",
          systemInstruction: "You are an automated primary care symptom analyzer. Output MUST strictly match the requested JSON schema. Include a general warning that this is automated guidance, and not a replacement for a professional diagnosis.",
        },
      });

      const bodyText = response.text?.trim() || "{}";
      try {
        const parsed = JSON.parse(bodyText);
        res.json(parsed);
      } catch {
        // Fallback clean-up if the model wrapped it in md tags
        const cleaned = bodyText.replace(/```json/i, "").replace(/```/g, "").trim();
        res.json(JSON.parse(cleaned));
      }
    } catch (error: any) {
      console.error("AI Analysis Error:", error);
      res.status(500).json({ error: error.message || "An error occurred during symptom analysis." });
    }
  });

  // REST API endpoint for Interactive Medical AI Chat supporting Images and Doctor Handoff Summaries
  app.post("/api/analyze-medical-chat", async (req, res) => {
    try {
      const { message, image, mimeType, history, isDoctorHandoff } = req.body;

      // 1. STRICT EMERGENCY CHECK FIRST
      const concatText = `${message || ""} ${history ? history.map((h: any) => h.text).join(" ") : ""}`.toLowerCase();
      const hasEmergencyKeywords = 
        concatText.includes("chest pain") || 
        concatText.includes("difficulty breathing") || 
        concatText.includes("breathing difficulty") || 
        concatText.includes("difficulty in breathing") || 
        concatText.includes("loss of consciousness") || 
        concatText.includes("unconscious") || 
        concatText.includes("stroke") || 
        concatText.includes("heavy bleeding") || 
        concatText.includes("sudden severe worsening") || 
        concatText.includes("severe chest pain");

      if (hasEmergencyKeywords) {
        return res.json({
          reply: "EMERGENCY ALERT: Seek immediate medical attention or contact emergency services."
        });
      }

      const ai = getAI();
      if (!ai) {
        // Fallback offline mock answer that satisfies the format patterns
        if (isDoctorHandoff) {
          return res.json({
            reply: `DOCTOR HANDOFF SUMMARY\nPatient Symptoms:\n- Reported symptoms: ${message || "N/A"}\n\nImage Findings:\n- Simulated offline view: No image can be analyzed in server offline mode.\n\nPossible Conditions:\n- Common Cold / Mild Viral Syndrome\n\nUrgency Level:\n- Low\n\nQuestions for Doctor:\n1. How long have they been feeling this way?\n2. Are they currently on any chronic medications?\n3. Any notable prior allergic reactions?`
          });
        }
        if (image) {
          return res.json({
            reply: `IMAGE ANALYSIS REPORT\nVisible Findings:\n- Image uploaded in offline mode (Simulated dermatitis or tissue inflammation).\n\nPossible Conditions:\n1. Contact Dermatitis — Confidence: 70%\n2. Eczema — Confidence: 40%\n\nSeverity:\n- Medium\n\nRecommended Specialty:\n- Dermatology Specialist\n\nRecommended Next Steps:\n- Keep the area clean and hydrated. Avoid strong soaps or allergens.\n\nDisclaimer:\n- This is AI-assisted guidance and not a confirmed medical diagnosis.`
          });
        }
        return res.json({
          reply: "I am currently running in a simulated offline assistant mode. Please consult in person if symptoms continue to worsen."
        });
      }

      // 2. Prepare Gemini Multimodal content components
      const parts: any[] = [];
      if (image && mimeType) {
        parts.push({
          inlineData: {
            mimeType: mimeType,
            data: image
          }
        });
      }

      // Add actual instruction/prompt as text part
      let medicalInstructionPrompt = "";
      if (isDoctorHandoff) {
        medicalInstructionPrompt = `Conduct a doctor handoff summary. Switch to doctor-assist mode. Summarize the patient's symptoms, image findings (if an image was uploaded), possible conditions, and level of urgency. Provide clinical handoff support clearly.
Provide exactly the following text structure format without exceptions:

DOCTOR HANDOFF SUMMARY
Patient Symptoms:
- [List symptoms reported]

Image Findings:
- [List image findings if any were present]

Possible Conditions:
- [List possible condition names]

Urgency Level:
- [Urgency Level]

Questions for Doctor:
1. [Question 1]
2. [Question 2]
3. [Question 3]`;
      } else if (image) {
        medicalInstructionPrompt = `Analyze the patient's uploaded image and provide the clinical report. Describe only visible findings. Do NOT claim a final diagnosis. State that this is AI-assisted guidance, not a medical diagnosis. Be professional, calm, and easy to understand.
Provide exactly the following text structure format:

IMAGE ANALYSIS REPORT
Visible Findings:
- [Finding 1]
- [Finding 2]

Possible Conditions:
1. [Condition] — Confidence: [Confidence]%
2. [Condition] — Confidence: [Confidence]%

Severity:
- [Low / Medium / High]

Recommended Specialty:
- [Specialty]

Recommended Next Steps:
- [Next step 1]
- [Next step 2]

Disclaimer:
- This is AI-assisted guidance and not a confirmed medical diagnosis.`;
      } else {
        medicalInstructionPrompt = `You are an AI Medical Assistant for a healthcare platform.
Review the patient's input: "${message || "Patient inquiry"}".
Ask relevant follow-up questions. Answer in simple language. Keep responses short and useful. Do not overstate certainty.
Let them know this is AI-assisted guidance, not a medical diagnosis.`;
      }

      parts.push({ text: medicalInstructionPrompt });

      // Support basic history context if available
      if (history && Array.isArray(history) && history.length > 0) {
        const historyText = history.slice(-4).map((h: any) => `${h.role === "user" ? "Patient" : "Assistant"}: ${h.text}`).join("\n");
        parts.push({ text: `Conversational Context history:\n${historyText}` });
      }

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: { parts },
        config: {
          systemInstruction: "You are a professional, calm, fact-based clinical AI assistant. You must respect the exact output formatting structures requested in the prompt based on image uploads or doctor handoff requests. Do not claim a final diagnosis. If emergency alert conditions are matched, raise alert."
        }
      });

      res.json({ reply: response.text || "No response received." });
    } catch (error: any) {
      console.error("AI Medical Chat Error:", error);
      res.status(500).json({ error: error.message || "An error occurred during interactive chat analysis." });
    }
  });

  // Configure Vite or Static delivery
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Healthcare Management Server running on port ${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start Healthcare Management server:", err);
});
