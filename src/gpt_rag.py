# gpt_rag.py

import os
import bs4
import tiktoken
import numpy as np
from langchain import hub
from typing import Callable, List, Dict, Any
from dotenv import load_dotenv
from operator import itemgetter
from langchain.schema import Document
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class GPTRAG:
    def __init__(self):
        self.sessionID = "agent_config.txt"  # Defines the file to store the context in
        self.load_environment()
        self.index_documents()
        self.setup_retriever()

    def load_environment(self):
        load_dotenv()  # Load and set the environment variables from the .env file
        os.environ['USER_AGENT']           = os.getenv('USER_AGENT', 'DefaultUserAgent/1.0')
        os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', '')
        os.environ['LANGCHAIN_ENDPOINT']   = os.getenv('LANGCHAIN_ENDPOINT', '')
        os.environ['LANGCHAIN_API_KEY']    = os.getenv('LANGCHAIN_API_KEY', '')
        os.environ['LANGCHAIN_PROJECT']    = os.getenv('LANGCHAIN_PROJECT', '')
        os.environ['OPENAI_API_KEY']       = os.getenv('OPENAI_API_KEY', '')

    def index_documents(self):
        # Load blog documents
        loader = WebBaseLoader(
            web_paths=(
                "https://github.com/THUDM/AgentBench/blob/main/README.md",
                "https://github.com/THUDM/AgentBench/blob/v0.1/docs/tutorial.md",
            )
        )
        blog_docs = loader.load()
        merged_docs = blog_docs

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, 
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(merged_docs)

        # Index documents
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings()
        )

    def setup_retriever(self):
        self.retriever = self.vectorstore.as_retriever()

    def multiquery(self, question: str) -> List[Document]:
        """Generate multiple queries and retrieve relevant documents."""
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives 
            | ChatOpenAI(temperature=0) 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        def get_unique_union(documents: list[list]):
            """Unique union of retrieved docs."""
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            unique_docs = list(set(flattened_docs))
            return [loads(doc) for doc in unique_docs]

        retrieval_chain = generate_queries | self.retriever.map() | get_unique_union
        retrieved_docs = retrieval_chain.invoke({"question": question})
        return retrieved_docs

    def ragfusion(self, question: str) -> List[Document]:
        """RAG-Fusion: Related."""
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion 
            | ChatOpenAI(temperature=0)
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        def reciprocal_rank_fusion(results: list[list], k=60):
            """Reciprocal_rank_fusion that takes multiple lists of ranked documents 
               and an optional parameter k used in the RRF formula."""
            fused_scores = {}
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    fused_scores[doc_str] += 1 / (rank + k)
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            return reranked_results

        retrieval_chain = generate_queries | self.retriever.map() | reciprocal_rank_fusion
        retrieved_docs = retrieval_chain.invoke({"question": question})
        return [doc for doc, score in retrieved_docs]

    def generate(self, question: str, model: ChatOpenAI, max_tokens: int = 512) -> str:
        """Generate an answer based on the question."""
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Choose a query constructor (multiquery or ragfusion)
        retrieval_chain = self.ragfusion  # You can switch to self.multiquery if desired
        retrieved_docs = retrieval_chain(question)

        # Combine retrieved documents into context
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Create the final chain
        final_rag_chain = (
            {"context": context, "question": question} 
            | prompt
            | model
            | StrOutputParser()
        )

        # Generate the final output
        gen = final_rag_chain.invoke({"question": question, "context": context})
        return gen

# # ======================================================================= #
# # =========================== INIT AND CONFIG =========================== #
# # ======================================================================= #
# import os
# import bs4
# import tiktoken
# import numpy as np
# from langchain import hub
# from typing import Callable
# from dotenv import load_dotenv
# from operator import itemgetter
# from langchain.schema import Document
# from langchain.load import dumps, loads
# from langchain.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# load_dotenv()  # Load and set the environment variables from the .env file
# os.environ['USER_AGENT']           = os.getenv('USER_AGENT')
# os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
# os.environ['LANGCHAIN_ENDPOINT']   = os.getenv('LANGCHAIN_ENDPOINT')
# os.environ['LANGCHAIN_API_KEY']    = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_PROJECT']    = os.getenv('LANGCHAIN_PROJECT')
# os.environ['OPENAI_API_KEY']       = os.getenv('OPENAI_API_KEY')



# # ======================================================================= #
# # ============================== INDEXING =============================== #
# # ======================================================================= #
    
# # Load blog
# loader = WebBaseLoader(
#     # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     web_paths=("https://github.com/THUDM/AgentBench/blob/main/README.md", 
#                "https://github.com/THUDM/AgentBench/blob/v0.1/docs/tutorial.md",)
#     # bs_kwargs=dict(
#     #     parse_only=bs4.SoupStrainer(
#     #         class_=("post-content", "post-title", "post-header")
#     #     )
#     # ),
# )
# blog_docs = loader.load()
# sessionID = "agent_config.txt" # defines the file to store the context in

# # # load convo
# # if os.path.exists(sessionID):
# #     print(f"Loaded conversation from {sessionID}.")
# #     with open(sessionID, "r") as f:
# #         ctx = "".join(f.readlines())
# # else:
# #     print("No conversation found, starting from scratch.")
# #     ctx = ""

# # ctx_doc = Document(page_content=ctx, metadata={"source": sessionID})
# # merged_docs = blog_docs + [ctx_doc]

# merged_docs = blog_docs

# # Split
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=300, 
#     chunk_overlap=50)

# # Make splits
# splits = text_splitter.split_documents(merged_docs)

# # Index
# vectorstore = Chroma.from_documents(documents=splits, 
#                                     embedding=OpenAIEmbeddings())

# retriever = vectorstore.as_retriever()

# # ======================================================================= #
# # ========================= QUERY CONSTRUCTION ========================== #
# # ======================================================================= #


# def multiquery():
#     # Multi Query: Different Perspectives
#     template = """You are an AI language model assistant. Your task is to generate five 
#     different versions of the given user question to retrieve relevant documents from a vector 
#     database. By generating multiple perspectives on the user question, your goal is to help
#     the user overcome some of the limitations of the distance-based similarity search. 
#     Provide these alternative questions separated by newlines. Original question: {question}"""

#     prompt_perspectives = ChatPromptTemplate.from_template(template)

#     generate_queries = (
#         prompt_perspectives 
#         | ChatOpenAI(temperature=0) 
#         | StrOutputParser() 
#         | (lambda x: x.split("\n"))
#     )

#     def get_unique_union(documents: list[list]):
#         """ Unique union of retrieved docs """
#         # Flatten list of lists, and convert each Document to string
#         flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
#         unique_docs = list(set(flattened_docs))
#         return [loads(doc) for doc in unique_docs]

#     # Retrieve
#     retrieval_chain = generate_queries | retriever.map() | get_unique_union
#     # docs = retrieval_chain.invoke({"question":question})
#     return retrieval_chain


# def ragfusion():
#     # RAG-Fusion: Related
#     template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
#     Generate multiple search queries related to: {question} \n
#     Output (4 queries):"""
#     prompt_rag_fusion = ChatPromptTemplate.from_template(template)

#     generate_queries = (
#         prompt_rag_fusion 
#         | ChatOpenAI(temperature=0)
#         | StrOutputParser() 
#         | (lambda x: x.split("\n"))
#     )

#     from langchain.load import dumps, loads

#     def reciprocal_rank_fusion(results: list[list], k=60):
#         """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
#             and an optional parameter k used in the RRF formula """
        
#         # Initialize a dictionary to hold fused scores for each unique document
#         fused_scores = {}

#         # Iterate through each list of ranked documents
#         for docs in results:
#             # Iterate through each document in the list, with its rank (position in the list)
#             for rank, doc in enumerate(docs):
#                 # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
#                 doc_str = dumps(doc)
#                 # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
#                 if doc_str not in fused_scores:
#                     fused_scores[doc_str] = 0
#                 # Retrieve the current score of the document, if any
#                 previous_score = fused_scores[doc_str]
#                 # Update the score of the document using the RRF formula: 1 / (rank + k)
#                 fused_scores[doc_str] += 1 / (rank + k)

#         # Sort the documents based on their fused scores in descending order to get the final reranked results
#         reranked_results = [
#             (loads(doc), score)
#             for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#         ]

#         # Return the reranked results as a list of tuples, each containing the document and its fused score
#         return reranked_results

#     retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
#     # docs = retrieval_chain.invoke({"question": question})
#     return retrieval_chain


# def something():
#     # Decomposition
#     template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
#     The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
#     Generate multiple search queries related to: {question} \n
#     Output (3 queries):"""
#     prompt_decomposition = ChatPromptTemplate.from_template(template)

#     # LLM
#     llm = ChatOpenAI(temperature=0)

#     # Chain
#     generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

#     # Run
#     question = "What are the main components of an LLM-powered autonomous agent system?"
#     questions = generate_queries_decomposition.invoke({"question":question})




#     # Prompt
#     template = """Here is the question you need to answer:

#     \n --- \n {question} \n --- \n

#     Here is any available background question + answer pairs:

#     \n --- \n {q_a_pairs} \n --- \n

#     Here is additional context relevant to the question: 

#     \n --- \n {context} \n --- \n

#     Use the above context and any background question + answer pairs to answer the question: \n {question}
#     """
#     decomposition_prompt = ChatPromptTemplate.from_template(template)

#     def format_qa_pair(question, answer):
#         formatted_string = ""
#         formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
#         return formatted_string.strip()

#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#     q_a_pairs = ""
#     for q in questions: # recursive answering
#         rag_chain = (
#         {"context": itemgetter("question") | retriever, 
#         "question": itemgetter("question"),
#         "q_a_pairs": itemgetter("q_a_pairs")} 
#         | decomposition_prompt
#         | llm
#         | StrOutputParser())

#         answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
#         q_a_pair = format_qa_pair(q,answer)
#         q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

#     return answer


# # ======================================================================= #
# # ============================= GENERATION ============================== #
# # ======================================================================= #


# # # python -m src.start_task -a
# # # python -m src.assigner




# def generate(question, model, max_tokens):
#     # RAG template
#     template = """Answer the following question based on this context:

#     {context}

#     Question: {question}
#     """
#     # Create the prompt from the template
#     prompt = ChatPromptTemplate.from_template(template)

#     # Execute the retrieval chain directly to get the context
#     retrieval_chain = multiquery()  # choose a query constructor
#     retrieved_docs = retrieval_chain.invoke({"question": question})

#     # Now, run the final chain with the retrieved context and the question
#     final_rag_chain = (
#         {"context": retrieved_docs,  # Pass the retrieved docs here
#          "question": question}  # Pass the original question
#         | prompt
#         | model
#         | StrOutputParser()
#     )

#     # Generate the final output
#     gen = final_rag_chain.invoke({"question": question})
#     return gen

# if __name__ == "__main__":
#     model = ChatOpenAI(temperature=0.4)
#     question = "How do I use a rag agent, ran and defined locally (within rag.py file) with the v0.2 version of AgentBench?" 
#     g = generate(question, model, 100)
#     with open(sessionID, "a+") as f:
#         f.write("\n=================================================\n")
#         f.write(g)
#     print(g)

