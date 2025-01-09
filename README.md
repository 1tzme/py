### README.md

# Chat with LLMs Models

This repository contains a simple Streamlit application for interacting with LLMs (Large Language Models) using the `Ollama` library. The app enables users to input queries and receive responses from a language model in real-time.

## Features
- Stream chat interface with real-time responses.
- User-friendly interface powered by Streamlit.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/1tzme/py
    cd streamlit-llm-chat
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate   # For Windows
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open the app in your browser at `http://localhost:8501`.

3. Select a model from the sidebar.

4. (Optional) Drag and drop or Browse files and press 'Load Documents'.

5. Enter your question and interact with assistant in real-time.
