import streamlit as st
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate


# ===== Helper: Extract text from PDF =====
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ===== Streamlit UI =====
st.set_page_config(page_title="LangExtract PDF", layout="wide")

st.title("📄 LangExtract: PDF Information Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.info("✅ PDF uploaded successfully!")

    # Extract text
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("🔍 Extracted PDF Text (first 500 chars)")
    st.text(pdf_text[:500])

    # ===== Define what info you want to extract =====
    response_schemas = [
        ResponseSchema(name="summary", description="A concise summary of the document"),
        ResponseSchema(name="keywords", description="Important keywords in a comma-separated list"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that extracts useful information from PDFs."),
        ("user", "Extract summary and keywords from the following text:\n\n{pdf_text}\n\n{format_instructions}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Button for extraction
    if st.button("Extract Info"):
        with st.spinner("Analyzing PDF..."):
            chain = prompt | llm | output_parser
            result = chain.invoke({
                "pdf_text": pdf_text[:2000],   # limit text if large
                "format_instructions": format_instructions
            })

            st.subheader("📌 Extracted Information")
            st.json(result)
