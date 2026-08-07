# 📚 Knowledge-Based RAG Chatbot

An AI-powered Knowledge-Based Retrieval-Augmented Generation (RAG) Chatbot that allows users to upload one or more PDF documents and ask questions based only on the uploaded content. The chatbot retrieves relevant information using Hybrid Retrieval (FAISS + BM25) and generates accurate responses using Groq Llama 3.1.

---

## Features

- Upload one or more PDF documents
- Ask questions based only on uploaded documents
- Hybrid Retrieval using FAISS and BM25
- HuggingFace MiniLM Embeddings
- Groq Llama 3.1 LLM
- Real-time response streaming
- Query preprocessing
- Regex-based answer cleaning
- Chat history support
- Simple and interactive Streamlit UI

---

## Screenshots

### Chatbot Response

![Chatbot](assets/screenshot1.png)

### Another Example

![Chatbot](assets/screenshot2.png)

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Frontend | Streamlit |
| Backend | Python |
| Framework | LangChain (LCEL) |
| LLM | Groq (Llama 3.1-8B Instant) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| PDF Processing | PyPDF |
| Environment Management | Python Dotenv |

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/SreyaBondili/Knowledge-Based-RAG-Chatbot.git
```

2. Open the project in VS Code or any Python IDE.

3. Install the required dependencies

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Groq API key

```env
GROQ_API_KEY=your_api_key
```

5. Run the application

```bash
streamlit run app.py
```

6. Open the application in your browser

```text
http://localhost:8501
```

7. Upload one or more PDF documents and start asking questions.

---

## 📂 Project Structure

```text
Knowledge-Based-RAG-Chatbot/
│
├── assets/
│   ├── screenshot1.png
│   └── screenshot2.png
│
├── app.py
├── requirements.txt
├── README.md
└── env_example.txt
```

---

## Key Highlights

- Hybrid Retrieval (FAISS + BM25)
- Semantic Search with FAISS
- Keyword Search using BM25
- Query Cleaning
- Regex-based Answer Cleaning
- Real-time Streaming Responses
- Modular LangChain Expression Language (LCEL) Pipeline

---

## Future Enhancements

- Support for Word, Excel, and PowerPoint documents
- User Authentication
- Persistent Chat History
- Multi-language Support
- Cloud Deployment
- Source Citation and Highlighting

---

## Author

**Sreya Bondili**

GitHub: https://github.com/SreyaBondili

---











