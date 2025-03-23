import streamlit as st
import pandas as pd
import torch
import gc
import torchaudio
import whisper
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import pyarrow.parquet as pq

api_key = st.secrets["API"]["API_KEY"]

# ----------------- 🔹 Free Up Memory Before Processing 🔹 -----------------
gc.collect()
torch.cuda.empty_cache()

# Check GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Using device: {DEVICE}")

# ----------------- 🔹 Streamlit UI Setup 🔹 -----------------
st.title("🎬 AI Subtitle Search Engine with RAG")

st.sidebar.header("Upload Your Audio")
uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

# ----------------- 🔹 Optimized Parquet Loading 🔹 -----------------
@st.cache_resource
def load_data_streaming(file_path, chunk_size=5000, max_rows=7000):
    """Efficiently loads Parquet file in chunks using PyArrow."""
    try:
        parquet_file = pq.ParquetFile(file_path)
        df_list, total_rows = [], 0

        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=["name", "cleaned_subtitles"]):
            df_chunk = batch.to_pandas()
            df_list.append(df_chunk)
            total_rows += len(df_chunk)
            if total_rows >= max_rows:
                break

        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading Parquet file: {e}")
        return pd.DataFrame()

# Load Parquet with optimized method
df = load_data_streaming("refined_cleaned_subtitles1.parquet", chunk_size=5000, max_rows=7000)

# Ensure DataFrame is not empty
if df.empty:
    st.error("No data found! Check the file path or format.")
    st.stop()
else:
    st.success(f"Loaded {len(df):,} rows successfully!")

# 🔹 Convert `cleaned_subtitles` to string and optimize memory usage
df["cleaned_subtitles"] = df["cleaned_subtitles"].astype(str).str.strip()
df["name"] = df["name"].astype("category")  # Reduce memory for categorical column

# ----------------- 🔹 Text Chunking with LangChain 🔹 -----------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)

# Process data in chunks to prevent high memory usage
documents = []
for text in df["cleaned_subtitles"].dropna().tolist():
    documents.extend([Document(page_content=t) for t in text_splitter.split_text(text)])

# Limit to avoid memory overload
MAX_DOCS = 3000
documents = documents[:MAX_DOCS]

# ----------------- 🔹 Embedding Model Setup 🔹 -----------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE}
)

# ----------------- 🔹 ChromaDB Vector Store 🔹 -----------------
CHROMA_DB_DIR = "./chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

if not os.listdir(CHROMA_DB_DIR):  # Load only if empty
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_DB_DIR)
else:
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)

retriever = vectorstore.as_retriever()

# ----------------- 🔹 Google Gemini AI Model 🔹 -----------------
chat_model = ChatGoogleGenerativeAI(model="gemini-pro", api_key=api_key)

# ----------------- 🔹 Chat Prompt for RAG 🔹 -----------------
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the following subtitles as context to answer the query."),
    MessagesPlaceholder(variable_name="context"),
    ("human", "Query: {question}"),
    ("ai", "Answer:")
])

# ----------------- 🔹 RetrievalQA Chain 🔹 -----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": chat_prompt, "document_variable_name": "context"},
    return_source_documents=True
)

# ----------------- 🔹 Whisper Transcription 🔹 -----------------
@st.cache_data
def transcribe_audio(audio_path):
    """Transcribes an audio file using Whisper 'tiny.en' model."""
    model = whisper.load_model("tiny.en", device=DEVICE)

    # Load & trim audio (1 min max)
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform[:, :sample_rate * 60]

    # Save trimmed audio temporarily
    temp_audio_path = "temp_audio.wav"
    torchaudio.save(temp_audio_path, waveform, sample_rate)

    return model.transcribe(temp_audio_path)["text"]

# ----------------- 🔹 User Query Processing 🔹 -----------------
if uploaded_audio:
    temp_audio_path = "temp_uploaded_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_audio.read())

    with st.spinner("Transcribing audio..."):
        audio_text = transcribe_audio(temp_audio_path)

    st.subheader("Transcribed Query")
    st.write(audio_text)

    with st.spinner("Retrieving relevant subtitles..."):
        response = qa_chain.invoke({"question": audio_text})

    # Display AI-generated response
    st.subheader("🔹 AI-Generated Response")
    st.write(response["result"])

    # Display Source Subtitles
    st.subheader("📜 Source Subtitles")
    for doc in response["source_documents"]:
        st.write(doc.page_content)