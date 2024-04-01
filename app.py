import streamlit as st 
import google.generativeai as genai 
import google.ai.generativelanguage as glm 
from dotenv import load_dotenv
from PIL import Image
import os 
import io 
import PyPDF2

load_dotenv()

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr

API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

gemini_pro, gemini_vision = st.columns(2)

def main():
    with gemini_pro:
        st.header("Interact with AICHAT Pro Vision")
        st.write("")

        prompt = st.text_input("Chat please...", placeholder="Prompt", label_visibility="visible")
        model = genai.GenerativeModel("Gemini-Pro")

        if st.button("ENTER",use_container_width=True):
            response = model.generate_content(prompt)

            st.write("")
            st.header(":blue[Response]")
            st.write("")

            st.markdown(response.text)

    with gemini_vision:
        st.header("Interact with Gemini Pro Vision")
        st.write("")

        input_type = st.radio("Input type", ("Text", "Image", "PDF"))

        if input_type == "Text":
            image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")

        elif input_type == "Image":
            image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
            uploaded_file = st.file_uploader("Choose an Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

            if uploaded_file is not None:
                st.image(Image.open(uploaded_file), use_column_width=True)

                st.markdown("""
                    <style>
                            img {
                                border-radius: 10px;
                            }
                    </style>
                    """, unsafe_allow_html=True)
                
        elif input_type == "PDF":
            pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if pdf_file is not None:
                pdf_reader = PyPDF2.PdfFileReader(pdf_file)
                text = ""
                for page_num in range(pdf_reader.numPages):
                    text += pdf_reader.getPage(page_num).extractText()
                image_prompt = st.text_area("Extracted text from PDF", text)

        if st.button("GET RESPONSE", use_container_width=True):
            model_name = "Gemini-Pro-Vision" if input_type == "Image" else "Gemini-Pro"
            model = genai.GenerativeModel(model_name)

            if input_type == "Text" or input_type == "PDF":
                if image_prompt != "":
                    response = model.generate_content(image_prompt)

                    st.write("")
                    st.write(":blue[Response]")
                    st.write("")

                    st.markdown(response.text)

                else:
                    st.write("")
                    st.header(":red[Please provide a message]")

            elif input_type == "Image":
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
                        st.header(":red[Please provide a message]")

                else:
                    st.write("")
                    st.header(":red[Please provide an image]")

if __name__ == "__main__":
    main()

