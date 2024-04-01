import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
from PIL import Image
import os
import io

load_dotenv()

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

gemini_pro, gemini_vision, ai_chat_tab = st.columns([1, 1, 2])

def main():
    with gemini_pro:
        st.header("Interact with Parvazbot Pro")
        st.write("")

        prompt = st.text_input("Chat please...", placeholder="Prompt", label_visibility="visible")
        model = genai.GenerativeModel("gemini-pro")

        if st.button("ENTER", use_container_width=True):
            response = model.generate_content(prompt)

            st.write("")
            st.header(":blue[Response]")
            st.write("")

            st.markdown(response.text)

    with gemini_vision:
        st.header("Interact with Gemini Pro Vision")
        st.write("")

        image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
        uploaded_file = st.file_uploader("Choose an Image", accept_multiple_files=False,
                                         type=["png", "jpg", "jpeg", "img", "webp"])

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
                            parts=[
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
                st.header(":red[Please Provide an image]")

    with ai_chat_tab:
        st.header("AI Chat")

        user_input = st.text_input("You:", "")
        if st.button("Send"):
            if user_input.strip() != "":
                model = genai.GenerativeModel("gemini-pro", temperature=0.3)
                response = model.generate_content(user_input)
                st.text_area("AI:", value=response.text, height=200)

if __name__ == "__main__":
    main()

