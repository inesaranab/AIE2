from openai import OpenAI
from dotenv import load_dotenv
import os

# Added for Grok
from xai_sdk import Client
from xai_sdk.chat import user, system

load_dotenv()


class ChatOpenAI:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")

    def run(self, messages, text_only: bool = True, **kwargs):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )

        if text_only:
            return response.choices[0].message.content

        return response


class ChatGrok:
    def __init__(self, model_name: str = "grok-3"):
        self.model_name = model_name
        self.grok_api_key = os.getenv("XAI_API_KEY")
        if self.grok_api_key is None:
            raise ValueError("XAI_API_KEY is not set")
        
        self.client = Client(api_key=self.grok_api_key)
        self.chat = None
    
    def run(self, messages, text_only: bool = True, **kwargs):
        """
        Run a chat completion with Grok.
        
        :param messages: List of message dictionaries with 'role' and 'content'
        :param text_only: If True, return only the text content
        :param kwargs: Additional parameters for the chat completion
        :return: Response content or full response object
        """
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        
        # Create a new chat session
        self.chat = self.client.chat.create(model=self.model_name)
        
        # Convert messages to xai_sdk format and append them
        for message in messages:
            if message["role"] == "user":
                self.chat.append(user(message["content"]))
            elif message["role"] == "system":
                self.chat.append(system(message["content"]))
            # Note: xai_sdk doesn't have assistant role, so we skip it
        
        # Get the response
        response = self.chat.sample()
        
        if text_only:
            return response.content
        
        return response
