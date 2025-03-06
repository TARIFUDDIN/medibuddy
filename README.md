# MediBot - Medical Question Answering System

MediBot is an AI-powered medical assistant that can answer health-related questions based on medical literature. The system uses natural language processing and a vector database to provide accurate medical information from reliable sources.

## Features

- Chat-based interface for asking medical questions
- Uses advanced language models for accurate responses
- Provides sources for all information presented
- Self-contained system that can run locally or be deployed to the cloud

## Requirements

- Python 3.8 or higher
- Streamlit
- LangChain
- HuggingFace account and API token
- FAISS for vector search

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/medibot.git
   cd medibot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your HuggingFace API token:
   - Create a `.streamlit/secrets.toml` file (this file should not be committed to Git)
   - Add your HuggingFace token:
     ```
     HF_TOKEN = "your-huggingface-token"
     ```

4. Add medical PDFs:
   - Create a `Data` directory
   - Place medical PDF documents in the `Data` directory

5. Index your documents:
   ```
   python creatememory.llm.py
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run medibot.py
   ```

2. Open your browser and navigate to the provided URL (usually http://localhost:8501)

3. Ask medical questions in the chat interface

## Structure

- `medibot.py` - Main Streamlit application file
- `connectwithllm.py` - Handles connections to LLM models
- `creatememory.llm.py` - Indexes documents for the vector database
- `requirements.txt` - Required Python packages
- `Data/` - Directory for storing medical PDFs
- `vectorstore/` - Generated vector database (not included in repository)

## Deployment

You can deploy this application to Streamlit Cloud, Hugging Face Spaces, or other Python-compatible platforms.

## Disclaimer

This application is for informational purposes only and not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.