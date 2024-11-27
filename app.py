import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image
import os 
import io 

# Load environment variables from .env file
load_dotenv()

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            if type(e)._name_ == "PdfReadError":
                st.warning(f"Skipping non-PDF file: {pdf.name}. Error: {str(e)}")
                continue
            else:
                raise
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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

def pdf_chat():
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

def ai_chat():
    st.header("Financial Analysis")

    prompt = st.text_input("Chat please...", placeholder="Prompt", label_visibility="visible")
    if st.button("ENTER", use_container_width=True):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        st.write("")
        st.header(":blue[Response]")
        st.write("")
        st.markdown(response.text)

    st.header("Interact with Gemini Pro Vision")
    st.write("")
    image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
    uploaded_file = st.file_uploader("Choose and Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), use_column_width=True)

        st.markdown("""
            <style>
                    img {
                        border-radius: 10px;
                    }
            </style>
            """, unsafe_allow_html=True)
        
    if st.button("GET RESPONSE", use_container_width=True):
        model = genai.GenerativeModel("gemini-pro-vision")

        if uploaded_file is not None:
            if image_prompt != "":
                image = Image.open(uploaded_file)

                response = model.generate_content(
                    glm.Content(
                        parts = [
                            glm.Part(text=image_prompt),
                            glm.Part(
                                inline_data=glm.Blob(
                                    mime_type="image/jpeg",
                                    data=image_to_byte_array(image)
                                )
                            )
                        ]
                    )
                )

                response.resolve()

                st.write("")
                st.write(":blue[Response]")
                st.write("")

                st.markdown(response.text)

            else:
                st.write("")
                st.header(":red[Please Provide a message]")

        else:
            st.write("")
            st

def main():
    st.title("AI reporting")

    pdf_chat_tab = st.container()
    ai_chat_tab = st.container()

    with pdf_chat_tab:
        pdf_chat()

    with ai_chat_tab:
        ai_chat()

if __name__ == "__main__":
    main()


