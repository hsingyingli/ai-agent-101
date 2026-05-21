from google.genai.client import Client
from pydantic import BaseModel
from typing import Literal

from google.genai import types


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str


history: list[Message] = []
tools_registry = {}
_ROLE_MAP = {"user": "user", "assistant": "model"}


class Agent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    client: Client
    tools: list = []

    def chat(self, user_prompt):
        history.append(Message(role="user", text=user_prompt))

        response = self.client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=self._message_to_contents(history),
        )

        reply_chunks: list[str] = []
        for chunk in response:
            if chunk.text:
                yield chunk.text
                reply_chunks.append(chunk.text)
        history.append(Message(role="assistant", text="".join(reply_chunks)))

    def _message_to_contents(self, messages: list[Message]):
        return [
            types.Content(
                role=_ROLE_MAP[msg.role],
                parts=[types.Part(text=msg.text)],
            )
            for msg in messages
        ]
