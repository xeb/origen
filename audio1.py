import os
import asyncio
import base64
import contextlib
import datetime
import json
import wave
import itertools
import asyncio
from google import genai
from google import genai
from google.genai import types
import contextlib
#from playsound import playsound
#import winsound

async def async_enumerate(it):
  n = 0
  async for item in it:
    yield n, item
    n +=1

# Read the GEMINI_API_KEY from ~/.ssh/gemini_api_key.txt
with open(os.path.expanduser("~/.ssh/gemini_api_key.txt"), "r") as f:
    GEMINI_API_KEY = f.read().strip()

client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
model_id = "gemini-2.0-flash-exp"
# config = {"response_modalities": ["TEXT"]}
config = {"response_modalities": ["AUDIO",]}

@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

async def main():
    async with client.aio.live.connect(model=model_id, config=config) as session:
        file_name = 'audio.wav'
        while True:
            message = input("User> ")
            if message.lower() == "exit":
                break

            with wave_file(file_name) as wav:
                await session.send(input=message, end_of_turn=True)

                turn = session.receive()
                async for n, response in async_enumerate(turn):
                    if response.data is not None:
                        wav.writeframes(response.data)

                        if n==0:
                            print(response.server_content.model_turn.parts[0].inline_data.mime_type)
                            print('.', end='')

            print("Playing...")
            #playsound(file_name)
            #winsound.PlaySound(file_name, winsound.SND_FILENAME)
            os.system(f"afplay {file_name}")

if __name__ == "__main__":
    asyncio.run(main())
