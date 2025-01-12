import asyncio
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)
import os

PROJECT_ID = "progeny-413722"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL_ID = "gemini-2.0-flash-exp"

config = LiveConnectConfig(response_modalities=["TEXT"])

async def main():
    async with client.aio.live.connect(
        model=MODEL_ID,
        config=config,
    ) as session:
        while True:
            message = input("User> ")
            if message.lower() == "exit":
                break
            await session.send(input=message, end_of_turn=True)

            async for response in session.receive():
                if response.text is None:
                    continue
                print(response.text, end="")

if __name__ == "__main__":
    asyncio.run(main())