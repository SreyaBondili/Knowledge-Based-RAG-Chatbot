📚 Knowledge-Based RAG Chatbot

Project Description
A Knowledge-Based Retrieval-Augmented Generation (RAG) Chatbot that allows users to upload one or more PDF documents and ask questions based only on the uploaded content. The chatbot retrieves the most relevant information using Hybrid Retrieval (FAISS + BM25) and generates accurate, context-aware responses using Groq Llama 3.1.

Features
•	Upload one or more PDF documents 
•	Ask questions based only on the uploaded documents 
•	Hybrid Retrieval using FAISS (Semantic Search) and BM25 (Keyword Search) 
•	HuggingFace MiniLM embeddings for semantic understanding 
•	Groq Llama 3.1 for fast response generation 
•	Query preprocessing for improved retrieval accuracy 
•	Regex-based answer cleaning to remove unnecessary filler phrases 
•	Chat history support for conversational context 
•	Real-time streaming responses (ChatGPT-like typing effect) 
•	Interactive user interface built with Streamlit

Chatbot Response
![Chat](assets/screenshot1.png)
![Chat](assets/screenshot2.png)

Technologies Used
•	Frontend                :       Streamlit 
•	Backend                 :       Python 
•	Framework               :       LangChain (LCEL) 
•	LLM                     :       Groq (Llama 3.1-8B Instant) 
•	Embeddings              :       HuggingFace (all-MiniLM-L6-v2) 
•	Vector Database         :       FAISS 
•	Keyword Retrieval       :       BM25 
•	PDF Processing          :       PyPDF 
•	Environment Management  :       Python Dotenv

How to Run
1.	Clone the repository:
git clone git clone https://github.com/SreyaBondili/Knowledge-Based-RAG-Chatbot.git  
2.	Open the project in VS Code or any Python IDE. 
3.	Install the required dependencies:
pip install -r requirements.txt
4.	Create a .env file and add your Groq API key:
GROQ_API_KEY=your_api_key
5.	Run the application:
streamlit run app.py
6.	Open http://localhost:8501 in your browser. 
7.	Upload one or more PDF files and start asking questions.

Project Structure
knowledge-rag-chatbot/
│
├── assets/
│   ├── screenshot1.png
│   └── screenshot2.png
│
├── app.py                  // Main application
├── requirements.txt        // Python dependencies
└── .env_example.txt        // Environment variable template

Key Highlights
•	Hybrid Retrieval for improved retrieval accuracy. 
•	Semantic search with FAISS and keyword matching with BM25. 
•	Query cleaning to ignore formatting instructions during retrieval. 
•	Answer cleaning using Regular Expressions (Regex). 
•	Real-time response streaming. 
•	Modular LangChain Expression Language (LCEL) pipeline.

 Future Enhancements
•	Support for Word, PowerPoint, and Excel documents 
•	User authentication 
•	Conversation history persistence 
•	Multi-language support 
•	Cloud deployment 
•	Citation and source highlighting







