import os
from dotenv import load_dotenv
# ------------------------------------------------------------------------
load_dotenv() # Load and set the environment variables from the .env file
os.environ['USER_AGENT'] = os.getenv('USER_AGENT')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
# ========================================================================

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#### INDEXING ####
# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = doc_splitter.split_documents(docs)

# Embed using TF-IDF
texts = [doc.page_content for doc in splits]
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(texts)

# Retrieval Function
def retrieve_relevant_documents(query, documents, embeddings, vectorizer):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    top_indices = ranked_indices[:5]  # Retrieve top 5 documents
    return [documents[i] for i in top_indices]

#### RETRIEVAL and GENERATION ####

# Prompt
def format_prompt(context, question):
    return f"""You are an AI assistant. Given the context below, provide a detailed yet concise answer to the question. Use clear and informative language. Don't include figures. Be concise.

Context:
{context}

Question:
{question}

Answer:"""

# LLM

# Load the tokenizer and model
print("Loading the model...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
class LlamaChain:
    def __init__(self, retriever, llm, tokenizer):
        self.retriever = retriever
        self.llm = llm
        self.tokenizer = tokenizer

    def invoke(self, question):
        # Retrieve context
        retrieved_docs = self.retriever(question)
        context = format_docs(retrieved_docs)
        output_start_index = len(context) # start printing after context

        # Prepare prompt
        full_prompt = format_prompt(context, question)
        inputs = self.tokenizer(full_prompt, return_tensors="pt")

        # Generate response
        output = self.llm.generate(**inputs, max_length=output_start_index+100, num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)[output_start_index:]

        return response

# Instantiate LlamaChain with custom retriever
rag_chain = LlamaChain(retriever=lambda q: retrieve_relevant_documents(q, splits, embeddings, vectorizer), llm=model, tokenizer=tokenizer)

# Question
response = rag_chain.invoke("What is Task Decomposition?")
print(response)