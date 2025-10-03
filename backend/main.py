import streamlit as st
import os
import traceback
from dotenv import load_dotenv

# Load env vars
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
##st.write("✅ GROQ_API_KEY loaded")

try:
    from pypdf import PdfReader
    from langchain_groq import ChatGroq   
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    from langchain.prompts import ChatPromptTemplate
except Exception as e:
    st.error(f"Import error: {e}")
    st.text(traceback.format_exc())
    st.stop()

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
st.set_page_config(page_title="LangExtract PDF JSON Extractor", layout="wide")
st.title("📄 LangExtract: Extract PDF Info as JSON")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.info("✅ PDF uploaded successfully!")
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("🔍 Extracted PDF Text (first 500 chars)")
    st.text(pdf_text[:500])

    # Define JSON schema for extraction
    response_schemas = [
        ResponseSchema(name="policy_number", description="Policy number"),
        ResponseSchema(name="effective_date", description="Policy effective date"),
        ResponseSchema(name="expiration_date", description="Policy expiration date"),
        ResponseSchema(name="insurer", description="Insurer details in JSON"),
        ResponseSchema(name="policyholder", description="Policyholder details in JSON"),
        ResponseSchema(name="coverage", description="Coverage details in JSON"),
        ResponseSchema(name="premium_and_payment", description="Premium and payment details"),
        ResponseSchema(name="policyholder_obligations", description="List of obligations"),
        ResponseSchema(name="claim_procedure", description="Claim procedure steps"),
        ResponseSchema(name="termination", description="Termination conditions")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that extracts structured JSON from insurance PDF documents."),
        ("user", "Extract all key fields from the following PDF text and return strictly in JSON:\n\n{pdf_text}\n\n{format_instructions}")
    ])

    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in environment. Please set it in your .env file.")
        st.stop()

    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",  
            temperature=0
        )
    except Exception as e:
        st.error(f"Groq LLM initialization error: {e}")
        st.text(traceback.format_exc())
        st.stop()

    if st.button("Extract JSON"):
        with st.spinner("Analyzing PDF and extracting JSON..."):
            try:
                chain = prompt | llm | output_parser
                result = chain.invoke({
                    "pdf_text": pdf_text[:3000],   # limit text to avoid overload
                    "format_instructions": format_instructions
                })

                st.subheader("📌 Extracted JSON Output")
                st.json(result)  # Show JSON nicely in Streamlit

            except Exception as e:
                st.error(f"Extraction error: {e}")
                st.text(traceback.format_exc())
else:
    st.info("Please upload a PDF file to begin.")
