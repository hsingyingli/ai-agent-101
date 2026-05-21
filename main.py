from google import genai
from src.settings import get_settings

settings = get_settings()

client = genai.Client(
    vertexai=True,
    project=settings.gcp_project,
    location=settings.gcp_location,
)


def main():
    response = client.models.generate_content_stream(
        model="gemini-3.1-flash-lite", contents="Explain how AI works in detail"
    )

    for chunk in response:
        print(chunk.text, end="", flush=True)


if __name__ == "__main__":
    main()
