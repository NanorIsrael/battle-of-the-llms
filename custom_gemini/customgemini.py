import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini model
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Create a class wrapper for compatibility
class LLMWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt):
        # Generate content using the Gemini model
        result = self.model.generate_content([prompt.text])
        # Assuming result is a list, and we need the first result's 'text' field
        if result:
            return result.candidates[0].content.parts
        return ""

# Instantiate the wrapper with the Gemini model
gemini_llm = LLMWrapper(gemini_model)

# Create a PromptTemplate
prompt_template = PromptTemplate(template="Translate the following English text to French: '{text}'")

# Create the RunnableSequence with the prompt template and the wrapped model
chain = RunnableSequence(prompt_template | gemini_llm)

# Define the input and get the response
# input_data = {"text": "Hello, how are you?"}
# response = chain.invoke(input_data)

# # Print the response
# print(response)
