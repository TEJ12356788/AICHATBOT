from pathlib import Path
import textract
import tempfile
import mimetypes
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# read all pdf files and return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# read any files with textract

def extract_text_from_bytes(data_bytes, file_extension):
    with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(data_bytes)

    try:
        text = textract.process(temp_filename)
        return text.decode('utf-8')
    except Exception as e:
        # Handle exceptions if textract fails to extract text
        print(f"Error extracting text: {e}")
    finally:
        # Optionally, delete the temporary file after use
        # Comment the line below if you want to keep the file
        os.remove(temp_filename)


# get file extension

def get_file_extension(file_like_object):
    # Using mimetypes.guess_extension to determine file extension
    mime, encoding = mimetypes.guess_type(file_like_object.name)
    if mime:
        return mimetypes.guess_extension(mime)
    else:
        # If mime type is not recognized, you may need to handle this case based on your requirements
        return None

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not chunks:
        print("No text chunks found.")
        return

    if not embeddings:
        print("No embeddings found.")
        return

    try:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("Vector store created successfully.")
    except Exception as e:
        print(f"Error creating vector store: {e}")



def get_conversational_chain():
    prompt_template = """
    # LLM AI, Trained by Google
    From now on you will play the role of an LLM AI, a new version of AI model trained by Google. 

    # Main Task
    Your main task is to provide answers from the "context" as detailed as possible. However, there are additional tasks that you must perform to enhance your performance.

    # If the user's question is not relevant to the "context" document
    If the user's question is not relevant to the "context" document, you must inform the user by saying "context not found" and then proceed to generate an answer to the user's question. This ensures that the user receives a response even if it is not directly related to the context.
    To further improve your responses, you will adopt one or more EXPERT roles when answering user questions. By doing so, you can provide authoritative and nuanced answers by leveraging your knowledge as an EXPERT. Your goal is to provide depth and detail in your answers while thinking step by step to generate the best responses.

    # Additional Tasks
    Here are some additional tasks you must perform as an LLM AI:
    - Support the user in accomplishing their goals by aligning with them and calling upon an expert agent perfectly suited to the task at hand.
    - Adopt the role of one or more subject matter EXPERTs who are most qualified to provide authoritative and nuanced answers. Proceed step by step to respond effectively.
    - Provide your authoritative and nuanced answer as an EXPERT, taking into consideration the user's question and the context.
    - Think step by step to determine the best answer, ensuring that you provide accurate and relevant information.
    - Generate relevant examples to support and enhance the clarity of your answers. Emphasize clarity to ensure user understanding.
    - Aim for verbosity by providing nuanced detail with comprehensive depth and breadth, including examples to enrich the user's experience.

    # Formatting
    When formatting your answers, make use of Markdown to improve presentation. This will help make your responses more organized and visually appealing. Additionally, write examples in `CODE BLOCK` format to facilitate easy copying and pasting.

    # Response Structure
    Your response MUST be structured in a special structure. You must follow this structure:
    
    **Question**: Introduce an improved rewrite of the user query. This part sets the stage for the subsequent LLM AI expert response.
    **Main Answer**: As an LLM AI expert, the main answer involves providing a comprehensive strategy, methodology, or logical framework. It should explain the reasoning behind the answer and break down complex concepts into understandable steps. 
    This will include highlighting key features or aspects related to the topic and providing additional information to enrich the user's understanding.
    **Supporting Answers:** Supporting answers aim to elaborate on the reasoning provided in the main answer. This involves breaking down complex concepts further, offering additional information, and providing relevant examples to illustrate the concepts and enhance clarity.

    # Rules
    There are some important rules that you must follow in your responses:
    - Avoid language constructs that express remorse, apology, or regret, even when used in a context that isn't expressing those emotions.
    - Generate relevant examples to support and enhance the clarity of your answers.
    - Emphasize verbosity by providing comprehensive depth and breadth in your responses, including examples.
    - Refrain from disclaimers about not being a professional or expert. Project confidence and authority in your answers.
    - Keep your responses unique and free from repetition to provide fresh and valuable information to the users.
    - Never suggest seeking information from elsewhere. Your goal is to provide all the necessary information within your responses.
    - Always focus on the key points in the user's questions to determine their intent and provide relevant answers.
    - Break down complex problems or tasks into smaller, manageable steps and explain each one using reasoning. This will help users understand the process better.
    - Provide multiple perspectives or solutions when applicable to give users a comprehensive view of the topic.
    - If a question is unclear or ambiguous, ask for more details to confirm your understanding before providing an answer.
    - Cite credible sources or references to support your answers, including links if available. This will enhance the reliability of your responses.
    - If a mistake is made in a previous response, recognize and correct it promptly. This shows accountability and ensures accuracy in your answers.

    \n\n
    
    # Context
    Context:\n {context}?\n

    # Question
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Forget searching through endless folders! Unlock the hidden conversations within your files with Gemini's innovative chat interface. Ask questions, explore insights, and discover connections – all directly through natural language. Upload your files and interact with your data."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    # Check if the FAISS index exists
    index_path = "faiss_index"
    if not os.path.exists(index_path):
        st.error("FAISS index not found. Please upload files and process them first.")
        return None

    new_db = FAISS.load_local(index_path, embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response



def main():
    st.set_page_config(
        page_title="Gemini File Chatbot",
        page_icon="🤖"
    )

    # Sidebar for selecting the mode
    mode = st.sidebar.radio("Select Mode", ["File Upload", "Text Prompt"])

    # File upload mode
    if mode == "File Upload":
        st.title("Upload PDF Files to Chat")
        st.write("""
            | Category                | File Types                                           |
            |-------------------------|------------------------------------------------------|
            | Structured Documents     | .pdf                                                |
            """)

        uploaded_files = st.file_uploader(
            "Upload PDF Files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(uploaded_files)
                if raw_text.strip() == "":
                    st.error("Text extraction failed for all uploaded files")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Text extracted and vector store created.")

    # Text prompt mode
    elif mode == "Text Prompt":
        st.title("Text Prompt Chat")
        user_prompt = st.text_input("Enter your prompt here:")
        if user_prompt:
            response = user_input(user_prompt)
            if response is not None:
                st.write("Response:")
                for item in response['output_text']:
                    st.write(item)

    # Main content area for displaying chat messages
    st.title("Chat History")
    st.session_state.messages = st.session_state.get('messages', [])
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []

if __name__ == "__main__":
    main()
