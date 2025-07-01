from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

# Step 1: Set the model
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)

# Step 2: Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate([('system', system_template), ('user', '{text}')])

# Step 3: StrOutputParser
parser = StrOutputParser()

# Step 4: Create chain
chain = prompt_template | model | parser

# App Definition
app = FastAPI(title="LangChain Server", version="1.0", description="A Simple API server using LangChain and LangServe")

# Adding Chain Routes
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)