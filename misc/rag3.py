# ======================================================================= #
# =========================== INIT AND CONFIG =========================== #
# ======================================================================= #
import os
import bs4
import tiktoken
import numpy as np
from langchain import hub
from typing import Literal
from dotenv import load_dotenv
from operator import itemgetter
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()  # Load and set the environment variables from the .env file
os.environ['USER_AGENT']           = os.getenv('USER_AGENT')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT']   = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY']    = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']    = os.getenv('LANGCHAIN_PROJECT')
os.environ['OPENAI_API_KEY']       = os.getenv('OPENAI_API_KEY')



# ======================================================================= #
# ============================== INDEXING =============================== #
# ======================================================================= #


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# LLM with function call 
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt 
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router 
router = prompt | structured_llm