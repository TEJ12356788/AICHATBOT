import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def main():
    st.title("Text and PDF Chat")

    # Function to extract text from PDF documents
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e:
                if type(e).__name__ == "PdfReadError":
                    st.warning(f"Skipping non-PDF file: {pdf.name}. Error: {str(e)}")
                    continue
                else:
                    raise  # Raise the exception if it's not a PdfReadError
        return text

    # Function for PDF chat functionality
    def pdf_chat():
        st.header("PDF Chat")

        # Option to upload PDF files or prompt a question
        option = st.radio("Choose an option", ["Upload PDF files", "Prompt a question"])

        if option == "Upload PDF files":
            # Allow user to upload PDF files
            pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    # Process the extracted text
                    st.success("PDF files processed successfully")

        elif option == "Prompt a question":
            # Allow user to input a question
            user_question = st.text_input("Ask a question about the PDF files")
            if user_question:
                if st.button("Ask"):
                    # Process the user question
                    st.write("Response:")

    # Function for AI chat functionality
    def ai_chat():
        st.header("AI Chat")

        user_input = st.text_input("You:", "")
        if st.button("Send"):
            if user_input.strip() != "":
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                response = model(user_input)
                st.text_area("AI:", value=response, height=200)

    # Display the PDF chat functionality
    pdf_chat()

    # Display the AI chat functionality
    ai_chat()

if __name__ == "__main__":
    main()

