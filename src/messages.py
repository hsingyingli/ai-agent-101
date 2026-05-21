from typing import Literal

from google.genai import types
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str


_ROLE_MAP = {"user": "user", "assistant": "model"}


def to_contents(messages: list[Message]) -> list[types.Content]:
    """Convert app-level chat history into google-genai Content objects.

    App uses "assistant" (industry convention); Gemini expects "model".
    """
    return [
        types.Content(
            role=_ROLE_MAP[msg.role],
            parts=[types.Part(text=msg.text)],
        )
        for msg in messages
    ]
