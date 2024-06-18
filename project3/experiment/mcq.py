import os
import json
import pandas as pd
import traceback
from langchain import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]
model = "google/flan-t5-base"

# Initialize HuggingFaceHub with your API token
hf_hub = HuggingFaceHub(
    repo_id="google/flan-t5-base",
)
