# From https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_starter.py
# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import io
import os
import json
import sys
import traceback

import cv2
import pyaudio
import wave
import tempfile
import PIL.Image
import mss

import argparse
from google import genai
from google.genai import types
import pyttsx3
from termcolor import colored

# Set the contents of ~/.ssh/gemini_api_key.txt to the env GOOGLE_API_KEY
with open(os.path.expanduser("~/.ssh/gemini_api_key.txt")) as f:
    os.environ["GOOGLE_API_KEY"] = f.read().strip()

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

USER_PROMPT = "User >"
ASSISTANT_PROMPT = "Assistant >"

MODEL = "models/gemini-2.0-flash-exp"
DEFAULT_MODE = "none"

client = genai.Client(http_options={"api_version": "v1alpha"})

function = dict(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
      "type": "OBJECT",
      "properties": {
          "location": {
              "type": "STRING",
              "description": "The city and state, e.g. San Francisco, CA",
          },
      },
      "required": ["location"],
    }
)

tool = types.Tool(function_declarations=[function])

class ToolWrapper():
    def __init__(self, functions_def):
        self.functions_def = functions_def

    @property
    def function_declarations(self):
        return self.functions_def

CONFIG = {
    "generation_config": {
        "response_modalities": ["TEXT"],
        "tools": [ ToolWrapper([function])  ], # This works (but is hacky)
        # "tools": [ tool ], # This errors with TypeError: Object of type Schema is not JSON serializable
        "system_instruction":
          [
            "You are a helpful Weather AI.",
            "Your mission is to see what the Current Weather is OR say something to the user if in doubt."
          ],
    }
}

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.text_buffer = []  # Buffer to accumulate text responses
        self.audio_stream = None
        self.output_stream = None
        # Initialize TTS engine directly instead of using TTSPlayer
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)


    async def convert_text_to_audio(self, text):
        """Convert text to audio using macOS say command directly to WAV"""
        try:
            # Create temporary file for WAV output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_filename = wav_file.name

            # Use afplay to generate speech directly to WAV
            process = await asyncio.create_subprocess_exec(
                'say',
                text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"Error running say command: {stderr.decode()}")
                return None

            # Use afplay to play the response directly
            process = await asyncio.create_subprocess_exec(
                'afplay',
                '/System/Library/Sounds/Glass.aiff',  # Or any other system sound
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()

            return b''  # Return empty bytes since we played the audio directly

        except Exception as e:
            print(f"Error converting text to audio: {e}")
            traceback.print_exc()
            return None


        except Exception as e:
            print(f"Error converting text to audio: {e}")
            traceback.print_exc()
            return None
        
        
    async def cleanup(self):
        """Safely clean up resources"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

    async def send_text(self):
        while True:
            try:
                text = await asyncio.to_thread(
                    input,
                    USER_PROMPT,
                )
                if text.lower() == "q" or text.lower() in ["quit", "exit"]:
                    break
                await self.session.send(input=text or ".", end_of_turn=True)
            except Exception as e:
                print(f"Error in send_text: {e}")
                break

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            while True:
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    break
                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)
        finally:
            if cap:
                cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]
        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        while True:
            try:
                frame = await asyncio.to_thread(self._get_screen)
                if frame is None:
                    break
                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)
            except Exception as e:
                print(f"Error in get_screen: {e}")
                break

    async def send_realtime(self):
        while True:
            try:
                msg = await self.out_queue.get()
                # print(f"\t\tSending {msg['mime_type']}")
                await self.session.send(input=msg)
            except Exception as e:
                print(f"Error in send_realtime: {e}")
                break

    async def listen_audio(self):
        try:
            mic_info = pya.get_default_input_device_info()
            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            
            if __debug__:
                kwargs = {"exception_on_overflow": False}
            else:
                kwargs = {}
                
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
        except Exception as e:
            print(f"Error in listen_audio: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()

    async def receive_responses(self):
        """Background task to read responses and handle both text and audio"""
        while True:
            try:
                turn = self.session.receive()
                has_introed = False
                current_response = []
                
                async for response in turn:

                    # Receive Multimodal Data (assume its Audio)
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)

                    # Receive Text
                    if text := response.text:
                        current_response.append(text)
                        if not has_introed:
                            print(f"\n{ASSISTANT_PROMPT} ", end="", flush=True)
                            has_introed = True
                        print(text, end="", flush=True)
                    
                    # Receive a Tool Call
                    elif ( response.server_content 
                        and response.server_content.model_turn 
                        and response.server_content.model_turn.parts 
                        ): # could be cleaner!
                        code_blocks = []
                        for part in response.server_content.model_turn.parts:
                            code_blocks.append(part.executable_code.code)

                        print(colored("\n\t" + str(code_blocks), "yellow"))
                
                # Process complete response
                if current_response:
                    complete_text = "".join(current_response)
                    print(f"\n{USER_PROMPT} ", end="", flush=True)
                    
                    # Convert text to audio and add to queue
                    pcm_data = await self.convert_text_to_audio(complete_text)
                    if pcm_data:
                        # Split PCM data into chunks to match expected audio format
                        chunk_size = CHUNK_SIZE * 2  # * 2 because we're using 16-bit audio
                        for i in range(0, len(pcm_data), chunk_size):
                            chunk = pcm_data[i:i + chunk_size]
                            if chunk:  # Only add non-empty chunks
                                self.audio_in_queue.put_nowait(chunk)
                    
                    current_response = []

                # Clear audio queue on interruption
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
                    
            except Exception as e:
                print(f"Error in receive_responses: {e}")
                break

    async def play_audio(self):
        try:
            self.output_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            while True:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(self.output_stream.write, bytestream)
        except Exception as e:
            print(f"Error in play_audio: {e}")
        finally:
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()

    async def run(self):
        try:
            # https://github.com/googleapis/python-genai/blob/main/google/genai/live.py#L629
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                async with asyncio.TaskGroup() as tg:
                    send_text_task = tg.create_task(self.send_text())
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    if self.video_mode == "camera":
                        tg.create_task(self.get_frames())
                    elif self.video_mode == "screen":
                        tg.create_task(self.get_screen())

                    tg.create_task(self.receive_responses())
                    tg.create_task(self.play_audio())

                    await send_text_task

        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\nShutting down...")
        except Exception as e:
            print(f"Error in run: {e}")
            traceback.print_exception(type(e), e, e.__traceback__)
        finally:
            await self.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pya.terminate()
