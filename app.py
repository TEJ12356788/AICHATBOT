import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import google.ai.generativelanguage as glm
import os

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

aichat_tab, aichat_vision_tab, pdf_chat_tab = st.columns([1, 1, 1])

def main():
    with aichat_tab:
        st.header("Interact with AICHAT Pro")
        st.write("")

        prompt = st.text_input("Chat please...", placeholder="Prompt", label_visibility="visible")
        model = genai.GenerativeModel("gemini-pro")

        if st.button("ENTER", use_container_width=True):
            response = model.generate_content(prompt)

            st.write("")
            st.header(":blue[Response]")
            st.write("")

            st.markdown(response.text)

    with aichat_vision_tab:
        st.header("Interact with AICHAT Pro Vision")
        st.write("")

        image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
        if st.button("GET RESPONSE", use_container_width=True):
            model = genai.GenerativeModel("gemini-pro-vision")

            if image_prompt != "":
                response = model.generate_content(image_prompt)

                st.write("")
                st.write(":blue[Response]")
                st.write("")

                st.markdown(response.text)
            else:
                st.write("")
                st.header(":red[Please provide a message]")

    with pdf_chat_tab:
        st.header("PDF Chat")

        option = st.radio("Choose an option", ["Upload PDF files", "Prompt a question"])

        if option == "Upload PDF files":
            pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF files processed successfully")

        elif option == "Prompt a question":
            user_question = st.text_input("Ask a question about the PDF files")
            if user_question:
                if st.button("Ask"):
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    st.write("Response:", response["output_text"])

if __name__ == "__main__":
    main()

