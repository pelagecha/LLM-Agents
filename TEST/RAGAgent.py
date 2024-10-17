# src/client/agents/RAGAgent.py

from typing import List, Dict
from src.agent import Agent
from src.gpt_rag import GPTRAG  # Correct absolute import
from langchain_openai import ChatOpenAI

class RAGAgent(Agent):
    """This agent uses Retrieval-Augmented Generation (RAG) to generate responses based on retrieved context."""

    def __init__(self, **kwargs) -> None:
        """
        Initialize the RAGAgent.

        Parameters:
            name (str): The name of the agent.
            body (dict): Additional parameters such as model and max_tokens.
        """
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'RAGAgent')
        body = kwargs.get('body', {})
        model_name = body.get('model', 'gpt-3.5-turbo')
        self.max_tokens = body.get('max_tokens', 512)

        # Initialize the GPTRAG instance
        self.gptrag = GPTRAG()

        # Initialize the language model
        self.model = ChatOpenAI(model_name=model_name, temperature=0.4, max_tokens=self.max_tokens)

    def inference(self, history: List[Dict[str, str]]) -> str:
        """
        Generate an inference based on the conversation history.

        Parameters:
            history (List[Dict[str, str]]): The conversation history.

        Returns:
            str: The generated response.
        """
        if not history:
            return "No history provided."

        # Extract the last user message
        last_message = history[-1]
        if last_message.get('role') != 'user':
            return "No user message to respond to."

        question = last_message.get('content', '')
        if not question:
            return "Empty question provided."

        # Generate the response using GPTRAG
        response = self.gptrag.generate(question=question, model=self.model, max_tokens=self.max_tokens)
        return response

# # # src/client/agents/RAGAgent.py
# from typing import List
# from src.agent import Agent

# class RAGAgent(Agent):
#     """This agent is a test agent, which does nothing. (return empty string for each action)"""

#     def __init__(self, **kwargs) -> None:
#         # load your model here
#         super().__init__(**kwargs)

#     def inference(self, history: List[dict]) -> str:
#         """Inference with the given history. History is composed of a list of dict, each dict:
#         {
#             "role": str, # the role of the message, "user" or "agent" only
#             "content": str, # the text of the message
#         }
#         """

#         # Finally return a string as the output
#         return "AAAAA"
    

# from gpt_rag import generate, CustomModel

# class RAGAgent:
#     def __init__(self, name, body=None):
#         self.name = name
#         self.max_tokens = body.get('max_tokens') if body else None
#         self.model = CustomModel()  # Use your custom model

#     def inference(self, history):
#         question = history[-1]['content']
#         return generate(question, max_tokens=self.max_tokens)
 