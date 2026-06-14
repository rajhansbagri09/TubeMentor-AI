# 🎬 TubeMentor AI

**TubeMentor AI** is an AI-powered YouTube Video Q&A Assistant built using **Retrieval-Augmented Generation (RAG)**. It allows users to chat with any YouTube video by extracting transcripts, creating embeddings, retrieving relevant context, and generating accurate answers using Large Language Models (LLMs).

---

## 🚀 Features

* 📺 Chat with YouTube videos using natural language
* 📝 Automatic transcript extraction from YouTube
* 🔍 Semantic search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 💬 Multi-turn conversational memory
* ⚡ MMR (Max Marginal Relevance) Retrieval
* 🎨 Interactive Streamlit user interface
* 🔄 Switch between videos using the Exit/New Video feature
* 🌍 Supports English, Hindi, and Hinglish transcripts

---

## 🏗️ How It Works

1. User provides a YouTube video URL.
2. Transcript is extracted using YouTube Transcript API.
3. Transcript is split into manageable chunks.
4. Chunks are converted into vector embeddings.
5. ChromaDB stores the embeddings.
6. User asks a question.
7. Retriever finds the most relevant transcript chunks.
8. Retrieved context is sent to the LLM.
9. TubeMentor AI generates an answer grounded in the video content.

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python
* LangChain

### Vector Database

* ChromaDB

### Embeddings

* BAAI/bge-m3 

### LLM

* Meta-Llama-3-8B-Instruct
* Hugging Face Inference API

### Data Source

* YouTube Transcript API

---

## 🌍 Language Support

TubeMentor AI supports:

* ✅ English Videos
* ✅ Hindi Videos
* ✅ Hinglish Videos
* ✅ Auto-generated YouTube Captions

Using multilingual embeddings allows semantic search across multiple languages while maintaining answer quality.

---

## 📂 Project Structure

```text
TubeMentor-AI/
│
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── chroma_db/

```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/rajhansbagri09/TubeMentor-AI.git
cd TubeMentor-AI
```

### Create Virtual Environment

```bash
conda create -n tubementor python=3.10
conda activate tubementor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Environment File

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Demo

### Home Page

Paste a YouTube URL and process the video.

### Video Processing

Transcript extraction, chunking, embedding generation, and vector storage.

### Chat Interface

Ask questions and receive answers based solely on the video content.

### Multi-Turn Conversation

Continue asking follow-up questions with maintained chat history.

---

## 🎯 Example Questions

* What is this video about?
* Summarize the video.
* Explain the main concepts discussed.
* What examples were mentioned?
* Give a detailed explanation of the second point.

---

## 🔒 Limitations

* Depends on the availability of YouTube transcripts.
* Cannot answer questions unrelated to the video content.
* Very long videos may take additional processing time.

---

## 🚀 Future Improvements

* Source citations for retrieved chunks
* Display video thumbnail and title
* Streaming responses
* PDF export of chat history
* Support for multiple videos simultaneously
* Cloud deployment
* Voice-based interaction

---

## 👨‍💻 Author

**Rajhans Bagri**

B.Tech CSE (Artificial Intelligence)

Aspiring AI Engineer | Machine Learning Enthusiast | RAG & Generative AI Developer

---

## ⭐ If you found this project useful

Please consider starring the repository and sharing your feedback.

---


