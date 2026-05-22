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
    max_tool_iteration: int = 5

    def chat(self, user_prompt):
        history.append(Message(role="user", text=user_prompt))
        contents = self._message_to_contents(history)

        for _ in range(self.max_tool_iteration):
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=contents,
                config=(self._tool_declarations()),
            )
            candidate_content = None
            function_calls = []
            if response.candidates:
                candidate_content = response.candidates[0].content
                if candidate_content:
                    parts = candidate_content.parts or []
                    function_calls = [p.function_call for p in parts if p.function_call]

            if not function_calls:
                text = response.text or ""
                history.append(Message(role="assistant", text=text))
                yield text
                return
            if candidate_content:
                contents.append(candidate_content)

            tool_response_parts = []
            for fc in function_calls:
                func = tools_registry[fc.name]["func"]
                result = func(**dict(fc.args or {}))
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            contents.append(types.Content(role="user", parts=tool_response_parts))

    def _tool_declarations(self):
        declarations = [tools_registry[f.__name__]["declaration"] for f in self.tools]
        if declarations:
            return types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=declarations)]
            )
        return None

    def _message_to_contents(self, messages: list[Message]):
        return [
            types.Content(
                role=_ROLE_MAP[msg.role],
                parts=[types.Part(text=msg.text)],
            )
            for msg in messages
        ]
