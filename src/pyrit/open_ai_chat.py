from openai import OpenAI
from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

class OpenAIChat(ChatSupport):
    def __init__(self, OPENAI_API_KEY) -> None:

        self.openai_client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
        self.temperature = 0
        self.top_p = 1
        self.max_tokens = 500
        self.seed = 100
        super().__init__()

    def complete_chat(self, *, messages: list[ChatMessage]) -> str:

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=self.max_tokens,
            seed=self.seed,
            temperature = self.temperature,
            top_p = self.top_p
        )

        return response.choices[0].message.content
  
