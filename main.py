from google import genai

from src.messages import Message, to_contents
from src.settings import get_settings

settings = get_settings()

client = genai.Client(
    vertexai=True,
    project=settings.gcp_project,
    location=settings.gcp_location,
)

history: list[Message] = []


def main():
    while True:
        user_input = input("> ")
        if not user_input.strip():
            continue
        if user_input.strip() == "/exit":
            print("Happy Coding!!")
            break

        history.append(Message(role="user", text=user_input))

        response = client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=to_contents(history),
        )

        reply_chunks: list[str] = []
        for chunk in response:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                reply_chunks.append(chunk.text)
        print()

        history.append(Message(role="assistant", text="".join(reply_chunks)))


if __name__ == "__main__":
    main()
