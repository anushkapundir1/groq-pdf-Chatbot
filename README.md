# 📄 Groq PDF Chatbot

## Overview
A Python-based LLM-powered chatbot that allows users to upload PDF documents and ask natural language questions.
The system uses LangChain to generate embeddings, performs semantic search, and returns contextually relevant answers.

## Features
- PDF upload and processing
- Text chunking and preprocessing
- Embedding generation
- Context-based Q&A using LLM


## How It Works
1. PDF files are uploaded and parsed into raw text
2. The text is split into smaller, meaningful chunks
3. Embeddings are generated for each chunk
4. Relevant chunks are retrieved using semantic similarity
5. An LLM generates a final response using retrieved context
   
PDF → Text Splitting → Embeddings → Vector Search → LLM → Answer

## Tech Stack
- Python
- LangChain
- Vector embeddings
- LLM API

## What I Learned
- End-to-end RAG pipeline
- Prompt engineering
- Embedding-based retrieval systems

## Future Improvements
- Add a UI (Streamlit / Flask)
- Support for multi-doc handling
