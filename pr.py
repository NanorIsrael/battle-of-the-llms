import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini model
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Import necessary modules for LLMChain
from langchain_core.runnables import Runnable, RunnableSequence
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create a class wrapper for compatibility with LLMChain
class LLMWrapper(Runnable):
    def __init__(self, model):
        self.model = model

    def invoke(self, prompt):
        # Generate content using the Gemini model
        result = self.model.generate_content([prompt])
        # Assuming result is a list, and we need the first result's 'text' field
        if result and 'text' in result[0]:
            return result[0]['text']
        return ""

    async def ainvoke(self, prompt):
        # For async compatibility, implement async version if needed
        return self.invoke(prompt)

# Instantiate the wrapper with the Gemini model
gemini_llm = LLMWrapper(gemini_model)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["language", "task"],
    template="Please write code in {language} to {task}."
)

# Create the LLMChain with the prompt template and the wrapped model
llm_chain = LLMChain(llm=gemini_llm.invoke, prompt=prompt_template)

# Define the input for the chain
input_data = {
    "language": "python",
    "task": "print hello"
}

# Invoke the chain and get the response
response = llm_chain(input_data)

# Print the response
print(response)
