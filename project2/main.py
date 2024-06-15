# from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"]
HUGGINGFACEHUB_API_TOKEN="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"
model = "google/flan-t5-base"


prompt = PromptTemplate(
	input_variables=["question"],
	template="Question: {question}\nAnswer:"
)

mytokenizer = AutoTokenizer.from_pretrained(model)
mymodel = AutoModelForSeq2SeqLM.from_pretrained(model, device_map='auto')

local_pipeline = pipeline(
	task="text2text-generation", 
	model=mymodel,
	tokenizer=mytokenizer,
	max_length=128,
)

question = "How many countries are in Africa?"
llm_chain = LLMChain(prompt=prompt, llm=HuggingFacePipeline(pipeline=local_pipeline))

response = llm_chain.run({"question": question})

print(response)
