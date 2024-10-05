# =========================== INIT AND IMPORT =========================== #
import os
import bs4
import numpy as np
import torch
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Callable
from langchain.schema import Document  # Ensure you have the correct import

load_dotenv()  # Load and set the environment variables from the .env file

# Set environment variables
os.environ['USER_AGENT']           = os.getenv('USER_AGENT')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_ENDPOINT']   = os.getenv('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY']    = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT']    = os.getenv('LANGCHAIN_PROJECT')

# =========================== DEVICE SETUP =========================== #
# Determine the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for acceleration.")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using Apple MPS for acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU. Consider upgrading your hardware for better performance.")

# =========================== DOCUMENT LOADING =========================== #
def load_documents(web_urls: List[str]) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=web_urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()

# =========================== TEXT SPLITTING =========================== #
def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return doc_splitter.split_documents(docs)

# =========================== EMBEDDING =========================== #
class Embedder:
    def __init__(self, method: str = 'tfidf'):
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        elif method == 'sentence_transformer':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            raise ValueError("Unsupported embedding method.")
        self.method = method

    def fit_transform(self, texts: List[str]):
        if self.method == 'tfidf':
            self.embeddings = self.vectorizer.fit_transform(texts)
        elif self.method == 'sentence_transformer':
            self.embeddings = self.model.encode(texts, convert_to_tensor=True)

    def transform(self, query: str):
        if self.method == 'tfidf':
            return self.vectorizer.transform([query])
        elif self.method == 'sentence_transformer':
            return self.model.encode([query], convert_to_tensor=True)

# =========================== RETRIEVAL =========================== #
def retrieve_relevant_documents(query: str, documents: List[Document], embedder: Embedder, top_k: int = 5) -> List[Document]:
    query_vec = embedder.transform(query)
    if embedder.method == 'tfidf':
        similarities = cosine_similarity(query_vec, embedder.embeddings).flatten()
        ranked_indices = np.argsort(similarities)[::-1]
    else:
        similarities = torch.nn.functional.cosine_similarity(torch.tensor(query_vec), torch.tensor(embedder.embeddings))
        ranked_indices = torch.argsort(similarities, descending=True)
    top_indices = ranked_indices[:top_k]
    return [documents[i] for i in top_indices]

# =========================== HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS) =========================== #
def generate_hypothetical_queries(original_query: str, num_hypotheses: int = 3) -> List[str]:
    # Placeholder for HyDE implementation
    # In practice, use a language model to generate hypothetical queries
    # For demonstration, we'll create simple variations of the original query
    hypotheses = [f"{original_query} in detail",
                  f"Explain {original_query}",
                  f"Provide an overview of {original_query}"]
    return hypotheses[:num_hypotheses]

# =========================== RAG FUSION =========================== #
def rag_fusion(retrieved_docs_list: List[List[Document]]) -> str:
    # Simple concatenation for demonstration
    # Advanced fusion strategies can be implemented as needed
    fused_context = "\n\n".join([doc.page_content for docs in retrieved_docs_list for doc in docs])
    return fused_context

# =========================== PROMPT FORMATTING =========================== #
def format_prompt(context: str, question: str) -> str:
    return f"""You are an AI assistant. Given the context below, provide a detailed yet concise answer to the question. Use clear and informative language. Don't include figures. Be concise. 

Context:
{context}

Question:
{question}

Answer:"""

# =========================== LANGUAGE MODEL SETUP =========================== #
class LanguageModel:
    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading the model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.model.eval()  # Set the model to evaluation mode

    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():  # Disable gradient computation for inference
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id  # To prevent warnings about no pad_token_id
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract the answer part
        answer = response[len(prompt):].strip()
        return answer

# =========================== LLM CHAIN =========================== #
class LlamaChain:
    def __init__(self, retriever: Callable[[str], List[Document]], llm: LanguageModel):
        self.retriever = retriever
        self.llm = llm

    def invoke(self, question: str) -> str:
        # First Layer: Generate Hypothetical Queries
        hypothetical_queries = generate_hypothetical_queries(question)

        # Retrieve documents for each hypothetical query
        retrieved_docs_list = [self.retriever(q) for q in hypothetical_queries]

        # Second Layer: RAG Fusion
        fused_context = rag_fusion(retrieved_docs_list)

        # Prepare prompt
        full_prompt = format_prompt(fused_context, question)

        # Generate response
        response = self.llm.generate_response(full_prompt)

        return response

# =========================== MAIN EXECUTION =========================== #
def main():
    # Load Documents
    web_urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/", "https://en.wikipedia.org/wiki/Leonardo_da_Vinci/"]
    docs = load_documents(web_urls)

    # Split Documents
    splits = split_documents(docs)

    # Verify Document Splits
    if not splits:
        print("No documents were split. Please check the document loader and splitter configurations.")
        return

    # Initialize Embedder (TF-IDF)
    embedder = Embedder(method='tfidf')
    texts = [doc.page_content for doc in splits]  # Corrected attribute access
    embedder.fit_transform(texts)

    # Initialize Language Model
    llm = LanguageModel(model_name="meta-llama/Llama-3.2-1B-Instruct", device=device)

    # Define Retriever with Multi-Query Capability
    def retriever(query: str) -> List[Document]:
        return retrieve_relevant_documents(query, splits, embedder, top_k=5)

    # Instantiate LlamaChain
    rag_chain = LlamaChain(
        retriever=retriever,
        llm=llm
    )

    # Question
    question = "What is Task Decomposition?"
    # question = "What did Davinci do?"
    response = rag_chain.invoke(question)
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()
