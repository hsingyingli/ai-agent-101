from google import genai

from src.agent import Agent
from src.tools import tool
from src.settings import get_settings

settings = get_settings()


@tool(description="when you need add two number")
def add(num1: int, num2: int) -> int:
    print(f"use tool to calculate {num1} + {num2}")
    return num1 + num2


agent = Agent(
    client=genai.Client(
        vertexai=True,
        project=settings.gcp_project,
        location=settings.gcp_location,
    ),
    tools=[add],
)


def main():
    while True:
        user_prompt = input("> ")
        if not user_prompt.strip():
            continue
        if user_prompt.strip() == "/exit":
            print("Happy Coding!!")
            break

        response = agent.chat(user_prompt)

        for chunk in response:
            print(chunk)


if __name__ == "__main__":
    main()
