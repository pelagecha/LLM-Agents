# ======================================================================= #
# =========================== INIT AND CONFIG =========================== #
# ======================================================================= #
import os
import bs4
import tiktoken
import numpy as np
from langchain import hub
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
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

# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

# Count tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
num_tokens_from_string(question, "cl100k_base")

# Text embedding models
from langchain_openai import OpenAIEmbeddings
embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
len(query_result)

# Cosine similarity 
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)

# Load blog (Indexing)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),)
blog_docs = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(blog_docs)



# ======================================================================= #
# ============================== RETRIEVAL ============================== #
# ======================================================================= #

# To index (VectorStores), create the vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # search arguments (k=1)
docs = retriever.invoke("What is Task Decomposition?") # used invoke because get_relevant_documents is deprecated


# ======================================================================= #
# ============================= GENERATION ============================== #
# ======================================================================= #

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Generate
gen = rag_chain.invoke("What is Task Decomposition?")
print(gen)


