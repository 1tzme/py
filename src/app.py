import os
import chromadb
import streamlit as st
from langchain_ollama import OllamaLLM
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_DIR = "uploaded_pdfs"
DB_DIR = "chroma_db"

os.makedirs(PDF_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="constitution_data",
    metadata={"description": "Kazakhstan Constitution"}
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = []
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            for line in page_text.split("\n"):
                if line.strip():
                    text.append(f"Page {i+1}: {line}")
        return text

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.readlines()
    return [f"TXT: {line.strip()}" for line in text if line.strip()]

def load_documents():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for file in os.listdir(PDF_DIR):
        file_path = os.path.join(PDF_DIR, file)
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file.endswith(".txt"):
            text = extract_text_from_txt(file_path)
        else:
            continue
        
        chunks = text_splitter.split_text("\n".join(text))
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"{file}_{i}"])

if not collection.peek():
    load_documents()

def query_chromadb(query_text, n_results=5):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    documents = results.get("documents", [])
    if documents and isinstance(documents[0], list):
        documents = [doc for sublist in documents for doc in sublist]
    return "\n\n".join(documents)

model_options = {"Ollama 3.2": "llama3.2", "Ollama 3.1:8b" : "llama3.1:8b"}
selected_model = st.sidebar.selectbox("Choose LLM Model", list(model_options.keys()))
llm_model = model_options[selected_model]

def query_ollama(prompt):
    llm = OllamaLLM(model=llm_model, host="http://localhost:11434")
    return llm.invoke(prompt)

st.title("AI Assistant: Constitution of Kazakhstan")

uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully! Reloading documents...")
    load_documents()

query_input = st.text_area("Enter your question about the Constitution:")
if st.button("Submit") and query_input.strip():
    context = query_chromadb(query_input)
    query_with_context = f"You are a professional legal expert specializing in the Constitution of Kazakhstan. Always cite the exact articles and their numbers when answering.\n\nContext: {context}\n\nQuestion: {query_input}" 
    response = query_ollama(query_with_context)

    # Добавляем в историю
    st.session_state.chat_history.append((query_input, response))

    st.write(f"**Q:** {query_input}")
    st.write(f"**A:** {response}")

if st.button("Show Chat History"):
    if st.session_state.chat_history:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.write(f"**{i+1}. Q:** {question}")
            st.write(f"**A:** {answer}")
            st.write("---")
    else:
        st.warning("No chat history found.")

if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
    st.success("Chat history cleared!")