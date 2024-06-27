import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os


load_dotenv()

def process_uploaded_image(uploaded_file):
	if uploaded_file is not None:
		image_data = uploaded_file.getvalue()
	
		image_parts = {
				'mime_type': uploaded_file.type,
				'data': image_data
			}
		
		return image_parts
	else:
		raise FileNotFoundError("No file uploaded")

input_prompt='''
Your ar an expert in understanding the details of images. You will
will recieve input images as invoices and you will have
to answer question based on the input image.
'''

st.set_page_config(page_title="Image Information Extractor")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg,", "png"])
image=""
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption="Uploaded Image.", width=600, use_column_width=False)

	user_query = st.text_input("Input Prompt", key="input")
	submit=st.button("Tell me about the image")

	if submit:
		image_parts = process_uploaded_image(uploaded_file)

		genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

		model = genai.GenerativeModel(model_name="gemini-1.5-flash")
		response = model.generate_content([input_prompt, image_parts, user_query])
		
		st.subheader("Response:")
		if not response:
			st.write("Loading...")
		st.write(response.text)
