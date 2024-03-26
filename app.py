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

    # Function to split text into smaller chunks
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    # Function to create and save FAISS vector store
    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    # Function to load and configure the conversational chain
    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    # Main function for PDF chat functionality
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
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF files processed successfully")

        elif option == "Prompt a question":
            # Allow user to input a question
            user_question = st.text_input("Ask a question about the PDF files")
            if user_question:
                if st.button("Ask"):
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    st.write("Response:", response["output_text"])

    # Function for AI chat functionality
    def ai_chat():
        st.header("AI Chat")
        user_input = st.text_input("You:", "")
        if st.button("Send"):
            if user_input.strip() != "":
                try:
                    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                    response = model(user_input)
                    st.text_area("AI:", value=response, height=200)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error(f"Input: {user_input}")
                    st.error(f"Model Configuration: model='gemini-pro', temperature=0.3")


    # Display the PDF chat functionality
    pdf_chat()

    # Display the AI chat functionality
    ai_chat()

if __name__ == "__main__":
    main()

