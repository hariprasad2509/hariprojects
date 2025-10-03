import streamlit as st
import os
import traceback
from dotenv import load_dotenv

# Load env vars
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Imports
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# Embeddings + FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===== Helper: Extract text from PDF =====
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        st.text(traceback.format_exc())
        return ""

# ===== Streamlit UI =====
st.set_page_config(page_title="LangExtract RAG Chatbot", layout="wide")
st.title("🤖 LangExtract: PDF Chatbot (RAG powered)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.info("✅ PDF uploaded successfully!")
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("🔍 Extracted PDF Text (first 500 chars)")
    st.text(pdf_text[:500])

    # ===== Split text into chunks =====
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(pdf_text)

    # ===== Create LangChain Documents =====
    documents = [Document(page_content=chunk) for chunk in chunks]

    # ===== Create Embeddings and FAISS Index =====
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # ===== Initialize Groq LLM =====
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment. Please set it in your .env file.")
        st.stop()

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # ===== Chat UI =====
    st.subheader("💬 Ask Questions about your PDF")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question from the PDF:")

    if st.button("Ask"):
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant docs
                    docs = vectorstore.similarity_search(user_question, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Build prompt
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant. Use the context to answer the question."),
                        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer in detail.")
                    ])

                    chain = prompt | llm
                    result = chain.invoke({"context": context, "question": user_question})

                    # Save chat history
                    st.session_state.chat_history.append(("User", user_question))
                    st.session_state.chat_history.append(("Bot", result.content))

                except Exception as e:
                    st.error(f"Chat error: {e}")
                    st.text(traceback.format_exc())

    # Show chat history
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        for role, msg in st.session_state.chat_history:
            if role == "User":
                st.markdown(f"**👤 {role}:** {msg}")
            else:
                st.markdown(f"**🤖 {role}:** {msg}")
else:
    st.info("Please upload a PDF file to begin.")
