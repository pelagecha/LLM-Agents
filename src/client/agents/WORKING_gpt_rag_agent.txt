from typing import List, Dict
from openai import OpenAI
from openai import OpenAIError  # Correct import for OpenAIError
from src.agent import Agent
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

client = OpenAI()
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

class GptRAGAgent(Agent):
    """This agent interacts with OpenAI's GPT-4 model."""

    # Define the supported roles
    SUPPORTED_ROLES = {'system', 'assistant', 'user', 'function', 'tool'}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def sanitize_role(self, role: str) -> str:
        """
        Sanitize and map roles to supported values.

        Args:
            role (str): The original role.

        Returns:
            str: A valid role supported by OpenAI.
        """
        role_mapping = {
            'agent': 'assistant',  # Map 'agent' to 'assistant'
            # Add other mappings if necessary
        }
        return role_mapping.get(role, role) if role in role_mapping else role

    def inference(self, history: List[Dict[str, str]]) -> str:
        """
        Perform inference with the given history.

        Args:
            history (List[Dict[str, str]]): A list of message dictionaries with "role" and "content".

        Returns:
            str: The assistant's response or an error message.
        """

        # Sanitize and map roles
        sanitized_history = []
        for message in history:
            original_role = message.get("role", "")
            sanitized_role = self.sanitize_role(original_role)
            if sanitized_role not in self.SUPPORTED_ROLES:
                return f"Invalid role detected in history: '{original_role}'. Supported roles are: {self.SUPPORTED_ROLES}"
            sanitized_history.append({
                "role": sanitized_role,
                "content": message.get("content", "")
            })

        # Build the messages list for GPT-4 from sanitized history
        messages = sanitized_history

        # Add a system message at the beginning if not already present
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        try:
            # Create a completion using OpenAI's GPT-4
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Ensure you're using a valid model name
                messages=messages,
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.2
            )

            # Extract and return the assistant's response
            choices = response.choices
            chat_completion = choices[0]
            content = chat_completion.message.content  # Correct for Chat Completions API
            return content

        except OpenAIError as e:
            # Handle OpenAI-specific errors
            return f"An error occurred with OpenAI: {e}"

        except Exception as e:
            # Handle any other unexpected errors
            return f"An unexpected error occurred: {e}"

'''
python eval.py --task configs/tasks/example.yaml --agent configs/agents/rag_agent.yaml --workers 1
'''