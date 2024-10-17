# from typing import List
# from src.agent import Agent
# from dotenv import load_dotenv
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# =========================== INIT AND IMPORT =========================== #
import os
import bs4
import numpy as np
import torch
from src.agent import Agent
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

class GptRAGAgent(Agent):
    def __init__(self, api_args=None, *args, **config):
        super().__init__(*args, **config)
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        # self.model_name = "meta-llama/Llama-2-7b-hf"
        # self.device = "cpu"
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using {self.device}")
        
        # Initialize Llama model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        # Initialize sentence transformer for embeddings
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        
        # Set default parameters
        self.max_length = config.get("max_length", 2048)
        self.temperature = config.get("temperature", 0.1)
        self.top_p = config.get("top_p", 0.95)
        
        # Set user agent for web requests
        default_user_agent = "default_user_agent"
        os.environ['USER_AGENT'] = os.getenv('USER_AGENT', default_user_agent)

    def load_and_process_documents(self, urls: List[str]) -> List[Document]:
        loader = WebBaseLoader(urls)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        return text_splitter.split_documents(documents)

    def get_relevant_documents(self, query: str, documents: List[Document], top_k: int = 1) -> List[Document]:
        query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=True)
        document_embeddings = self.sentence_transformer.encode([doc.page_content for doc in documents], convert_to_tensor=True)
        
        similarities = cosine_similarity(query_embedding.cpu().numpy(), document_embeddings.cpu().numpy())[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [documents[i] for i in top_indices]

    def generate_answer(self, query: str, relevant_docs: List[Document]) -> str:
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def inference(self, history: List[dict]) -> str:
        # Extract the last user message as the query
        query = next(msg['content'] for msg in reversed(history) if msg['role'] == 'user')
        
        # For simplicity, let's use a fixed URL. In a real scenario, you might want to
        # dynamically determine the URLs based on the query or conversation history.
        urls = ["https://en.wikipedia.org/wiki/Addition"]
        
        documents = self.load_and_process_documents(urls)
        relevant_docs = self.get_relevant_documents(query, documents)

        raw_answer = self.generate_answer(query, relevant_docs)
        answer = self.remove_context(raw_answer)
        print(self.device)
        return answer

    def remove_context(self, raw_answer: str) -> str:
        # !!!!!! HACK ONLY WORKS FOR NUMBERS Find method that doesn't break if "Answer: " is a part of the response !!!!!!!
        prefix = "Answer: "
        prefix_start = raw_answer.find(prefix)
        temp = raw_answer[prefix_start + len(prefix):].split()[0].strip()
        # temp = "".join([i for i in temp if isnum(i)])
        print(temp)
        return temp


    def format_prompt(self, history: List[dict]) -> str:
        formatted_prompt = ""
        for message in history:
            if message["role"] == "user":
                formatted_prompt += f"Human: {message['content']}\n"
            else:
                formatted_prompt += f"Assistant: {message['content']}\n"
        formatted_prompt += "Assistant:"
        return formatted_prompt

# from typing import List, Dict
# from openai import OpenAI
# from openai import OpenAIError  # Correct import for OpenAIError
# from src.agent import Agent
# from dotenv import load_dotenv
# import os

# # Load environment variables from the .env file
# load_dotenv()

# client = OpenAI()
# OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

# class GptRAGAgent(Agent):
#     """This agent interacts with OpenAI's GPT-4 model."""

#     # Define the supported roles
#     SUPPORTED_ROLES = {'system', 'assistant', 'user', 'function', 'tool'}

#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#     def sanitize_role(self, role: str) -> str:
#         """
#         Sanitize and map roles to supported values.

#         Args:
#             role (str): The original role.

#         Returns:
#             str: A valid role supported by OpenAI.
#         """
#         role_mapping = {
#             'agent': 'assistant',  # Map 'agent' to 'assistant'
#             # Add other mappings if necessary
#         }
#         return role_mapping.get(role, role) if role in role_mapping else role

#     def inference(self, history: List[Dict[str, str]]) -> str:
#         """
#         Perform inference with the given history.

#         Args:
#             history (List[Dict[str, str]]): A list of message dictionaries with "role" and "content".

#         Returns:
#             str: The assistant's response or an error message.
#         """

#         # Sanitize and map roles
#         sanitized_history = []
#         for message in history:
#             original_role = message.get("role", "")
#             sanitized_role = self.sanitize_role(original_role)
#             if sanitized_role not in self.SUPPORTED_ROLES:
#                 return f"Invalid role detected in history: '{original_role}'. Supported roles are: {self.SUPPORTED_ROLES}"
#             sanitized_history.append({
#                 "role": sanitized_role,
#                 "content": message.get("content", "")
#             })

#         # Build the messages list for GPT-4 from sanitized history
#         messages = sanitized_history

#         # Add a system message at the beginning if not already present
#         if not any(msg["role"] == "system" for msg in messages):
#             messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

#         try:
#             # Create a completion using OpenAI's GPT-4
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",  # Ensure you're using a valid model name
#                 messages=messages,
#                 max_tokens=100,
#                 n=1,
#                 stop=None,
#                 temperature=0.2
#             )

#             # Extract and return the assistant's response
#             choices = response.choices
#             chat_completion = choices[0]
#             content = chat_completion.message.content  # Correct for Chat Completions API
#             return content

#         except OpenAIError as e:
#             # Handle OpenAI-specific errors
#             return f"An error occurred with OpenAI: {e}"

#         except Exception as e:
#             # Handle any other unexpected errors
#             return f"An unexpected error occurred: {e}"

# '''
# python eval.py --task configs/tasks/example.yaml --agent configs/agents/rag_agent.yaml --workers 1
# '''