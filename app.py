import streamlit as st
import ollama
from pymongo import MongoClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlas
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import os

# Подключение к MongoDB Atlas
MONGO_URI = "mongodb+srv://your_username:your_password@your_cluster.mongodb.net/"
DB_NAME = "chatbot"
COLLECTION_NAME = "vector_store"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Настройка векторного хранилища
embedding = OpenAIEmbeddings()
vector_store = MongoDBAtlas(collection, embedding)

def load_constitution():
    url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = text_splitter.split_text(text)
        
        vector_store.add_texts(documents)
        st.success("Конституция успешно загружена и проиндексирована!")
    else:
        st.error("Ошибка загрузки Конституции")

st.title("AI Assistant for the Constitution of Kazakhstan")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Загрузить Конституцию Казахстана"):
    load_constitution()

llm = ChatOllama(model="llama3.2")
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

def generate_queries(user_query):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Переформулируй этот вопрос разными способами:"},
            {"role": "user", "content": user_query}
        ]
    )
    return response["message"]["content"].split("\n")

user_input = st.text_input("You:", "", key="user_input")

if user_input:
    queries = generate_queries(user_input)
    responses = [qa_chain.run(query) for query in queries]
    bot_response = "\n".join(responses)
    
    collection.insert_one({"role": "user", "content": user_input})
    collection.insert_one({"role": "bot", "content": bot_response})
    
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

for role, text in st.session_state.chat_history:
    st.text(f"{role}: {text}")
