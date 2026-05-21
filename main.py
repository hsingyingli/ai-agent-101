from google import genai

from src.agent import Agent
from src.settings import get_settings

settings = get_settings()


agent = Agent(
    client=genai.Client(
        vertexai=True,
        project=settings.gcp_project,
        location=settings.gcp_location,
    )
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
