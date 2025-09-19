# 🧠 Virtual Psychologist – Real-Time Conversational Mental Health Support

## 📖 Overview  
The **Virtual Psychologist** is an **AI-powered real-time conversational system** designed to provide accessible and personalized mental health support.  
It enables **natural, voice-based conversations** between users and an empathetic AI assistant capable of **understanding**, **analyzing**, and **responding in real time**.

The system is built to:

- **Predict potential mental illnesses** using a fine-tuned deep learning model.  
- **Detect AI response style** (calm, supportive, motivational, etc.) to guide how the model should respond to a user query.  
- **Recognize user intent** to understand their needs and context.  
- **Leverage Retrieval-Augmented Generation (RAG)** with **threshold-based checks** for accurate, relevant responses.  
- **Perform query expansion** to clarify vague user queries using conversation history.  
- **Dynamically adapt prompts** depending on the detected intent for better personalization.  
- **Provide a seamless real-time voice experience** using **Whisper** for speech-to-text and **Kokoro** for text-to-speech.  
- **Run only the necessary models** through an **AI Agent** to optimize performance and reduce latency.  
- **Ensure privacy and security** by encrypting and anonymizing all user data.

---

## 🎯 Problem Statement  
Mental health issues are on the rise, yet **affordable and immediate psychological support** is out of reach for many.  

**Virtual Psychologist** bridges this gap by providing:  
- **24/7 conversational support** with real-time voice interaction.  
- **Personalized therapy suggestions** based on illness prediction and emotional state.  
- **Safe, private, and judgment-free space** for users to express their feelings.  
- **Adaptive responses** tailored to the user’s mental state and needs.

---

## ⚙️ Features  
- 🎤 **Real-Time Voice Conversations** – Users talk to the system naturally instead of typing.  
- 🧪 **Illness Prediction** – Detects possible mental health issues such as depression, anxiety, or PTSD.  
- 💬 **AI Response Style Detection** – Determines how the AI should respond (e.g., calm, supportive, or motivational).  
- 🎯 **Intent Recognition** – Understands user intentions such as therapy seeking, casual talk, or session termination.  
- 🧠 **Empathetic LLM Responses** – Generates personalized, context-aware replies.  
- 📚 **RAG with Threshold Checking** – Ensures retrieved knowledge is relevant and above a similarity threshold before using it in responses.  
- 🌱 **Query Expansion** – Refines vague queries using past conversation context.  
- 📝 **Dynamic Prompting** – Adjusts the prompt based on detected user intent for better alignment and personalization.  
- 📊 **Memory Management** – Supports both **short-term session memory** and **long-term memory embeddings** for personalized conversations.  
- ⚡ **AI Agent Optimization** – Runs illness detection, response style detection, and intent recognition models **only when necessary**, improving performance.  
- 🌐 **Scalable & Dockerized Deployment** – Ready for cloud-based deployment with Docker.  
- 🔒 **Secure & Private** – Encrypted data handling with anonymization to protect user identity.

---

## 🏗️ System Architecture

User (Voice Input)
↓
Speech-to-Text (Whisper)
↓
AI Agent
┣ Illness Prediction (Fine-tuned DeBERTa-v3)
┣ AI Response Style Detection (Fine-tuned RoBERTa)
┗ Intent Recognition (Meta-Llama via OpenRouter)
↓
Dynamic Prompting Engine
↓
RAG Pipeline (Threshold Check + Query Expansion)
↓
Response Generation (Mistral via OpenRouter)
↓
Text-to-Speech (Kokoro)
↓
User (Voice Output)


---

## 🧩 Core Modules

### **1. Illness Prediction**
- **Goal:** Detect possible mental health conditions such as depression, anxiety, or PTSD.  
- **Model Used:** Fine-tuned [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base)  
  - Custom architectural modifications were applied to improve performance on mental health datasets.  
- **Output:** Predicted illness category used to guide therapy suggestions and response personalization.

---

### **2. AI Response Style Detection**
- **Goal:** Detect the **style of response** the AI should use (calm, supportive, motivational, etc.).  
- **Model Used:** Fine-tuned [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)  
- **Purpose:** Helps the LLM craft responses that match the user’s emotional needs.

---

### **3. Intent Recognition**
- **Goal:** Understand user intent to provide relevant guidance.
- **Examples:**
  - *"I need help coping with stress"* → Therapy Guidance  
  - *"I feel lonely"* → Supportive Conversation  
  - *"End session"* → Closing intent  
- **Model Used:** Meta Llama via **OpenRouter API**.

---

### **4. RAG with Threshold Checking**
- **Goal:** Enhance LLM responses by pulling external resources only when relevant.  
- **Features:**
  - **Threshold Checking:** Ensures that retrieved content meets a similarity threshold before being included in the response.  
  - **Query Expansion:** Refines vague queries using **conversation history embeddings** for better search precision.  
  - **Dynamic Prompting:** Prompts are adjusted dynamically based on the detected intent.  

**Example Workflow:**
User: "I'm anxious."
→ Expanded Query: "What are coping techniques for anxiety during stressful events?"
→ Threshold Check Passed
→ RAG fetches CBT techniques and relevant articles.



---

### **5. Memory Management**
- **Short-Term Memory:**  
  - **LangChain-Memoery** stores ongoing session data for conversational continuity.  
- **Long-Term Memory:**  
  - **PostgreSQL + pgvector** stores embeddings of past conversations.  
  - Enables **RAG** and deeper personalization.

---

### **6. Speech-to-Text (STT) & Text-to-Speech (TTS)**
- **STT:** [Whisper](https://openai.com/research/whisper)  
  Converts user voice input into accurate text for processing.  
- **TTS:** [Kokoro](https://github.com/kokoro-ai/kokoro)  
  Generates natural, human-like voice responses.

---

### **7. AI Agent Optimization**
The **AI Agent** intelligently determines **when to run certain models**:
- Runs illness prediction, response style detection, and intent recognition **only when required**, reducing unnecessary computation.

**Example:**
- *Casual chat* → Skip illness prediction.  
- *Critical support query* → Run all detection models.

---

## 🚀 Tech Stack

| **Component**         | **Technology** |
|-----------------------|----------------|
| **Backend Framework** | FastAPI |
| **Illness Prediction** | Fine-tuned DeBERTa-v3 |
| **AI Response Style Detection** | Fine-tuned DistilRoBERTa |
| **Intent Recognition** | Meta Llama (OpenRouter API) |
| **Final Response Generation** | Mistral (OpenRouter API) |
| **Speech-to-Text** | Whisper |
| **Text-to-Speech** | Kokoro |
| **Memory Management** | Redis, PostgreSQL + pgvector |
| **RAG Implementation** | Custom pipeline with threshold checks |
| **Deployment** | Docker |
| **Frontend (Future)** | React.js + WebRTC |

---

## 🐳 Running the Project (Dockerized Deployment)

### **1. Clone the Repository**
```bash
git clone https://github.com/muhamedamil/AI-Driven-Personalized-Therapy-Recommendations-system.git
cd AI-Driven-Personalized-Therapy-Recommendations-system


## 🚀 Deployment with Docker

Follow the steps below to build and run the application using **Docker**:

---

### **2. Build the Docker Image**

```bash
docker build -t virtual-psychologist .


docker run -d -p 8000:8000 virtual-psychologist


