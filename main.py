import streamlit as st
import tempfile
from utils import load_and_split_pdf
from rag_chat import create_vector_store, build_qa_chain

st.set_page_config(page_title="📚 PDF Q&A Chatbot")

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document..."):
        docs = load_and_split_pdf(tmp_path)
        vectorstore = create_vector_store(docs)
        qa_chain = build_qa_chain(vectorstore)
        st.success("Ready to chat!")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        result = qa_chain(query)
        st.markdown(f"**Answer:** {result['result']}")
