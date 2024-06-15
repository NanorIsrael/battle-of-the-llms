from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"]
model = "google/flan-t5-base"

# Initialize HuggingFaceHub with your API token
hf_hub = HuggingFaceHub(
    repo_id="google/flan-t5-base",
)

# Define a simple prompt template
prompt = PromptTemplate(input_variables=["question"], template="Question: {question}\nAnswer:")

# Create an LLMChain with the Hugging Face model
llm_chain = LLMChain(prompt=prompt, llm=hf_hub)

# Invoke the model with a question
question = "How many countries are in Africa?"
response = llm_chain.run({"question": question})

print(response)
