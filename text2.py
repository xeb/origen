import os
import asyncio
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)

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
        text_input = "Hello? Gemini are you there?"
        print(f"**Input:** {text_input}")

        await session.send(input=text_input, end_of_turn=True)

        response = []

        async for message in session.receive():
            if message.text:
                response.append(message.text)

        print(f"**Response >** {''.join(response)}")

if __name__ == "__main__":
    asyncio.run(main())