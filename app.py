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

    # Function to initialize the AI chat model
    def initialize_chat_model():
        return ChatGoogleGenerativeAI(model="gemini-pro")

    # Main function for text chat
    def text_chat():
        st.header("Text Chat")

        # Initialize the AI chat model
        chat_model = initialize_chat_model()

        # Text input for user question
        user_question = st.text_input("Ask a question")

        if user_question:
            if st.button("Ask"):
                response = chat_model.send_message(user_question, stream=True)
                st.write("Response:", response.text)

    # Main function for PDF chat functionality
    def pdf_chat():
        # Your existing PDF chat code here...

    # Display the text chat functionality
    text_chat()

    # Display the PDF chat functionality
    pdf_chat()

if __name__ == "__main__":
    main()

