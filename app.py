import streamlit as st
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ✅ Load Groq API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("📚 Groq PDF Chatbot (Latest LangChain)")
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    st.write("📥 Loading PDF...")

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    st.write("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)


    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer ONLY using PDF content.
    If info not present, reply: Not in PDF.

    Context:
    {context}

    Question: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

    st.success("✅ PDF processed successfully! Ask anything 👇")

    question = st.text_input("Ask a question from the PDF:")

    if question:
        with st.spinner("🔍 Searching document..."):
            answer = chain.invoke(question)

        st.write("### ✅ Answer:")
        st.write(answer.content)
