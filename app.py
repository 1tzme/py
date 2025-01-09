import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'documents' not in st.session_state:
    st.session_state.documents = ""

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def load_documents(uploaded_files):
    try:
        documents_content = ""
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read().decode("utf-8")
            documents_content += f"\n{file_content}"
        st.session_state.documents = documents_content
        st.success("Documents uploaded successfully!")
        logging.info("Documents loaded.")
    except Exception as e:
        st.error("Failed to load documents.")
        logging.error(f"Error loading documents: {str(e)}")

def main():
    st.title("Chat with LLMs Ollama")
    logging.info("App started")

    model = st.sidebar.selectbox("Choose a model", ["mymodel", "llama3.1:8b"])
    logging.info(f"Model selected: {model}")

    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .txt files", type=["txt"], accept_multiple_files=True
    )

    if st.sidebar.button("Load Documents") and uploaded_files:
        load_documents(uploaded_files)

    if st.session_state.documents:
        with st.expander("Loaded Documents Content"):
            st.text_area("Documents Content", st.session_state.documents, height=200, disabled=True)

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        if st.session_state.documents:
                            system_message = ChatMessage(
                                role="system",
                                content=f"The following context is from the uploaded documents: {st.session_state.documents}"
                            )
                            messages = [system_message] + [
                                ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages
                            ]
                        else:
                            messages = [
                                ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages
                            ]

                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()