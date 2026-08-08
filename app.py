import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq

# Loads variables from your .env file (e.g. GROQ_API_KEY) into the environment
load_dotenv()

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss" not in st.session_state:
    st.session_state.faiss = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "current_files" not in st.session_state:
    st.session_state.current_files = None

st.set_page_config(
    page_title="Knowledge-Based RAG Chatbot",
    page_icon="📚",
    layout="wide"
)
st.title("📚 Knowledge-Based RAG Chatbot")

# Read each stored message from chat history and display it in the chat window
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask anything about your document..."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

# Load and cache the Hugging Face embedding model for converting text into vectors
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Load and cache the Llama LLM through Groq for generating answers
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant"
    )

# PDF Processing Pipeline
def process_uploaded_files(uploaded_files):
    """
    Takes the list of files the user uploaded via Streamlit's file_uploader,
    loads their text content, splits it into chunks, and builds TWO
    retrievers from those chunks: one semantic (FAISS) and one keyword-based
    (BM25). Together they power hybrid_retrieve() later.
    """
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())
        os.remove(tmp.name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    faiss_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4  # also return top 4 results
    return faiss_retriever, bm25

# Trigger Processing When Files Are Uploaded
if uploaded_files:
    uploaded_names = tuple(file.name for file in uploaded_files)
    if (
        st.session_state.faiss is None
        or st.session_state.current_files != uploaded_names
    ):
        with st.spinner("Processing PDFs..."):
            faiss, bm25 = process_uploaded_files(uploaded_files)
            st.session_state.faiss = faiss
            st.session_state.bm25 = bm25
            st.session_state.current_files = uploaded_names
            st.session_state.chat_history = []
        st.sidebar.success("Documents processed successfully!")

# Regex patterns to detect answer format instructions from the user's query
FORMAT_INSTRUCTION_PATTERNS = [
    r"\bin \d+\s*(lines?|words?|sentences?|points?)\b",
    r"\bin (one|two|three|four|five|a few|short)\s*(lines?|words?|sentences?|points?)\b",
    r"\b(briefly|concisely|in short|in brief|in detail)\b",
]

def clean_query_for_retrieval(question: str):
    """
    Removes formatting/length instructions from the user's question so the
    retriever searches based on the actual CONTENT being asked about, not
    incidental phrasing like "explain in 2 lines".
    """
    cleaned = question
    for pattern in FORMAT_INSTRUCTION_PATTERNS:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE
        )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else question

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "to",
    "and", "define", "definition", "explain", "describe", "tell",
    "me", "about", "give", "for", "please", "with", "list",
    "line", "lines", "word", "words", "short", "brief", "briefly"
}

def keyword_overlap_score(text, keywords):
    """
    Counts how many of the query's meaningful keywords appear in a given
    chunk of text. Used to re-rank retrieved chunks so the most keyword-
    relevant ones are prioritized.
    """
    text_words = set(
        re.findall(
            r"[a-zA-Z]+",
            text.lower()
        )
    )
    return len(text_words & keywords)  # set intersection size

def hybrid_retrieve(question, faiss_retriever, bm25_retriever):
    """
    The core retrieval function. Takes the user's question PLUS the two
    retriever objects (passed in explicitly rather than pulled from
    st.session_state), and returns a single combined
    string of the most relevant document chunks.

    IMPORTANT: faiss_retriever and bm25_retriever are passed in as
    ARGUMENTS instead of being read from st.session_state inside this
    function. This is because LangChain's LCEL chain runs this function
    in a background thread (via RunnablePassthrough.assign), and
    Streamlit's st.session_state is only accessible from the main
    script-run thread. Reading session_state from a worker thread throws
    an AttributeError, even though the value clearly exists. Passing the
    retrievers in as plain Python arguments sidesteps that entirely.
    """

    search_query = clean_query_for_retrieval(question)
    faiss_results = faiss_retriever.invoke(search_query)
    bm25_results = bm25_retriever.invoke(search_query)
    results = []
    seen = set()
    for doc in faiss_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            results.append(doc)

    # Extract meaningful keywords from the query (excluding stopwords)
    keywords = set(
        re.findall(
            r"[a-zA-Z]+",
            search_query.lower()
        )
    ) - STOPWORDS

    if keywords:
        results.sort(
            key=lambda d:
            keyword_overlap_score(
                d.page_content,
                keywords
            ),
            reverse=True
        )
    results = results[:6]
    return "\n\n".join(
        doc.page_content
        for doc in results
    )

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY using the supplied context.

If multiple chunks contain relevant information,
combine them naturally.

Do not make up facts.

If the answer cannot be found in the context,
reply exactly:

"I don't know based on the provided document."

Never mention things like:
- Based on the context
- According to the document
- The document says
- I found

Just answer directly.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
""")

# Remove filler sentences from model output
FILLER_PATTERNS = [
    r"^based on the (provided |given |supplied )?(context|document)s?,?\s*",
    r"^according to the (provided |given |supplied )?(context|document)s?,?\s*",
    r"^the (provided |given |supplied )?(context|document)\s*(states|says|mentions|shows)\s*",
    r"^i found\s*",
    r"^the answer is:?\s*"
]

def clean_answer(text):
    """
    Repeatedly strips any leading filler phrase from the model's answer
    until none of the FILLER_PATTERNS match anymore (handles cases where
    multiple filler phrases are stacked at the start, e.g.
    "Based on the context, I found that..."), then capitalizes the first
    letter of whatever remains.
    """
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in FILLER_PATTERNS:
            new_cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE
            ).strip()

            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned

# Load the LLM (once, cached, safe at global scope)
llm = load_llm()

# Chat Handling
if question:
    if st.session_state.faiss is None:
        st.warning("Please upload at least one PDF.")
        st.stop()
    faiss_retriever = st.session_state.faiss
    bm25_retriever = st.session_state.bm25
    chat_history = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.chat_history
    )

    # Store user's query in session state to maintain chat history
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()  # a UI slot we can repeatedly overwrite
        answer = ""
        retrieval_chain = RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: hybrid_retrieve(x["question"], faiss_retriever, bm25_retriever)
            )
        )

        # LCEL (LangChain Expression Language) pipeline: retrieval -> prompt -> LLM
        chain = retrieval_chain | prompt | llm

        response = chain.stream(
            {
                "history": chat_history,
                "question": question
            }
        )

        # Chat Streaming
        for chunk in response:
            if chunk.content:
                answer += chunk.content

                placeholder.markdown(answer + "▌")
        answer = clean_answer(answer)
        placeholder.markdown(answer)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.divider()
