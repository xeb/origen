# -*- coding: utf-8 -*-
# -------------
# origen
# -------------
#
# This was initially copied from / inspired by: From https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_starter.py
# There are notable changes, namely CLI arguments for multiple input streams
# and Tool Use / Function Calling
#
# $ python live_cli.py --help
# usage: live_cli.py [-h] [--inputs [INPUTS ...]] [--output {text,audio,text_tts}] [--tools] [--audio_input {0}] [--debug]
# options:
#  -h, --help            show this help message and exit
# --inputs [INPUTS ...]
#                        List of inputs. Can be specified multiple times, comma-separated, or space-separated
#  --output {text,audio,text_tts}
#                        Output mode. Note that output modes cannot be mixed, unlike inputs. Text_TTS is just text output with a system TTS post process, primarily for
#                        demo purposes.
#  --tools               Whether or not to use Tools in the live stream
#  --audio_input {0}     The microphone you want to use.
#  --debug               Debug (right now) will print the 'inputs' being streamed
#
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

# TODO:
# (1) save screenshots when bounding oxes are identified
# (2) save logfile of activity

import asyncio
import base64
import io
import os
import sys
import traceback
import cv2
import pyaudio
import tempfile
import PIL.Image
import PIL.ImageDraw
import mss
import pyautogui
import asyncio
from threading import Event
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import argparse
from google import genai
from google.genai import types
from termcolor import colored
import Quartz
import AppKit

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

client = genai.Client(http_options={"api_version": "v1alpha"})

move_and_click_def = types.FunctionDeclaration(
      name="self.moveToThenClick",
      description="Moves the mouse to a location and then clicks that object",
      parameters= types.Schema(
        type="OBJECT",
        properties={
            "x1": types.Schema(
                type="INTEGER",
                description="The x1 coordinate to move to and click",
                ),
            "y1": types.Schema(
                type="INTEGER",
                description="The y1 coordinate to move to and click",
                ),
            "x2": types.Schema(
                type="INTEGER",
                description="The x2 coordinate to move to and click",
                ),
            "y2": types.Schema(
                type="INTEGER",
                description="The y2 coordinate to move to and click",
                ),
        },
        required=["x1", "y1", "x2", "y2"],),
    )

draw_box = types.FunctionDeclaration(
      name="self.draw_bounding_box",
      description="Draws a bounding box around an area of the screen",
      parameters= types.Schema(
        type="OBJECT",
        properties={
            "x1": types.Schema(
                type="INTEGER",
                description="The x1 coordinate to move to and click",
                ),
            "y1": types.Schema(
                type="INTEGER",
                description="The y1 coordinate to move to and click",
                ),
            "x2": types.Schema(
                type="INTEGER",
                description="The x2 coordinate to move to and click",
                ),
            "y2": types.Schema(
                type="INTEGER",
                description="The y2 coordinate to move to and click",
                ),
        },
        required=["x1", "y1", "x2", "y2"],),
    )

type_def = types.FunctionDeclaration(
      name="pyautogui.write",
      description="Write text on the screen",
      parameters= types.Schema(
        type="OBJECT",
        properties={
            "message": types.Schema(
                type="STRING",
                description="The text to write",
                ),
        },
        required=["message"],),
    )

press_def = types.FunctionDeclaration(
      name="pyautogui.press",
      description="Press a key on the keyboard, such as 'enter' or anything supported by pyautogui",
      parameters= types.Schema(
        type="OBJECT",
        properties={
            "keys": types.Schema(
                type="STRING",
                description="The key to press",
                ),
        },
        required=["keys"],),
    )

# tool = types.Tool(function_declarations=[move_and_click_def, type_def, press_def, ])
tool = types.Tool(function_declarations=[draw_box])

class SyncFunctionRunner():
    def __init__(self, context=None):
        self.context = context
        pass

    def exec(self, cmds: list[str]):
        print(colored("\n\tSync Function Runner: Processing " + str(cmds), "yellow"))
        for cmd in cmds:
            try:
                new_cmd = cmd.replace("default_api", "self")
                if new_cmd.startswith("self.pyautogui"):
                    new_cmd = new_cmd.replace("self.", "")
                if new_cmd.startswith("print_on_screen.self.print_on_screen"):
                    new_cmd = new_cmd.replace("print_on_screen.self.print_on_screen", "self.print_on_screen")
                if new_cmd.startswith("print_on_screen.print_on_screen("):
                    new_cmd = new_cmd.replace("print_on_screen.print_on_screen(", "self.print_on_screen(")

                exec(new_cmd)

            except Exception as e:
                print(colored(f"\n\tFunction Runner: Error executing command: {new_cmd}\n{e}", "red"))

    def print_on_screen(self, response):
        print(colored(f"\n\n{response}", "green"))

    def write(self, text):
        print(colored(f"\n\n{text}", "yellow"))

    # def moveToThenClick(self, x1, y1, x2, y2):
    #     print(colored(f"\n\tmoveToThenClick invoked to {x1=}{y1=}{x2=}{y2=}, but need to normalize with {self.context=}", "yellow"))
    #     x1_norm = int((x1 / 1000.0) * self.context["last_screen_width"])
    #     x2_norm = int((x2 / 1000.0) * self.context["last_screen_width"])
    #     y1_norm = int((y1 / 1000.0) * self.context["last_screen_height"])
    #     y2_norm = int((y2 / 1000.0) * self.context["last_screen_height"])
    #     print(colored(f"\n\tmoveToThenClick going to normalized to {x1_norm=}{y1_norm=}{x2_norm=}{y2_norm=}", "yellow"))

    #     # Get center x coordinate by averaging x1 and x2
    #     center_x = int((x1_norm + x2_norm) / 2)

    #     # Get center y coordinate by averaging y1 and y2
    #     center_y = int((y1_norm + y2_norm) / 2)

    #     print(colored(f"\n\tmoveToThenClick: Clicking on {center_x=},{center_y=}", "yellow"))
    #     # pyautogui.moveTo(x=center_x, y=center_y)
    #     pyautogui.click(x=center_x, y=center_y)

    def draw_bounding_box(self, x1, y1, x2, y2):
        try:
            # Get the last screen path from self
            last_screen_path = self.context["last_screen_path"]

            # Open the image using PIL
            img = PIL.Image.open(last_screen_path)
            
            x1_norm = int((x1 / 1000.0) * self.context["last_screen_width"])
            x2_norm = int((x2 / 1000.0) * self.context["last_screen_width"])
            y1_norm = int((y1 / 1000.0) * self.context["last_screen_height"])
            y2_norm = int((y2 / 1000.0) * self.context["last_screen_height"])

            # Create a draw object
            draw = PIL.ImageDraw.Draw(img)

            # Draw a red rectangle using the given coordinates
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3) # Increased width for visibility
            draw.rectangle((x1_norm, y1_norm, x2_norm, y2_norm), outline="blue", width=3) # Increased width for visibility

            # Save the image with the bounding box
            img.save("last_screen_withbox.jpg", format="jpeg")

        except Exception as e:
            print(colored(f"\n\tdraw_bounding_box: Error drawing or saving bounding box: {e}", "red"))


    def moveToThenClick(self, x1, y1, x2, y2):

        # Let's save the bounding box
        self.draw_bounding_box(x1, y1, x2, y2)

        print(colored(f"\n\tmoveToThenClick invoked to {x1=}{y1=}{x2=}{y2=}, but need to normalize with {self.context=}", "yellow"))

        # Get center x coordinate by averaging x1 and x2
        center_x = int((x1 + x2) / 2)

        # Get center y coordinate by averaging y1 and y2
        center_y = int((y1 + y2) / 2)

        print(colored(f"\n\tmoveToThenClick: Clicking on {center_x=},{center_y=}", "yellow"))
        pyautogui.click(x=center_x, y=center_y)


def list_input_devices():
    p = pyaudio.PyAudio()
    available_inputs = []
    print("Available input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        # Check if the device has input channels
        if device_info['maxInputChannels'] > 0:
            print(f"Device {i}: {device_info['name']}")
            print(f"  Input channels: {device_info['maxInputChannels']}")
            print(f"  Sample rate: {int(device_info['defaultSampleRate'])} Hz")
            available_inputs.append(i)
    
    p.terminate()
    return available_inputs

available_audio_inputs = list_input_devices()

pya = pyaudio.PyAudio()

class StreamLoop:
    def __init__(self, config=None, inputs=None, tts=False, debug=False, audio_input=1):
        self.config = config
        self.debug = debug
        self.inputs = inputs
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.text_buffer = []  # Buffer to accumulate text responses
        self.audio_stream = None
        self.output_stream = None
        self.tts = tts
        self.audio_input = audio_input

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

    async def send_text(self):
        while True:
            try:
                text = await asyncio.to_thread(
                    input,
                    USER_PROMPT,
                )
                if text.lower() == "q" or text.lower() in ["quit", "exit"]:
                    print("Quitting time")
                    sys.exit(0)
                    return
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

        img.save("last_camera.jpg", format="jpeg")

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "source": "camera", "data": base64.b64encode(image_bytes).decode()}

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

        self.last_screen_width = img.size[0]
        self.last_screen_height = img.size[1]

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        img.save("last_screen.jpg", format="jpeg")

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "source": "screen", "data": base64.b64encode(image_bytes).decode()}

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

                if self.debug:
                    print(f"\t\tSending {msg['mime_type']=} {msg['source']=}")
                msg.pop("source") # don't send to the API
                await self.session.send(input=msg)
            except Exception as e:
                print(f"Error in send_realtime: {e}")
                break

    async def listen_audio(self):
        try:
            mic_info = pya.get_device_info_by_index(self.audio_input)
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
                await self.out_queue.put({"source": "microphone", "data": data, "mime_type": "audio/pcm"})
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
                            if part.executable_code and part.executable_code.code:
                                code_blocks.append(part.executable_code.code)

                        if len(code_blocks) > 0:
                            func_run = SyncFunctionRunner({
                                "last_screen_width": self.last_screen_width,
                                "last_screen_height": self.last_screen_height,
                                "last_screen_path" : "last_screen.jpg"
                            })
                            try:
                                func_run.exec(code_blocks)
                            except Exception as e:
                                print(f"Error in function execution: {e}")

                
                # Process complete response
                if current_response:
                    complete_text = "".join(current_response)
                    print(f"\n{USER_PROMPT} ", end="", flush=True)
                    
                    # Convert text to audio and add to queue
                    if self.tts:
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
            # https://github.com/googleapis/python-genai/issues/121
            # config can be types.LiveConnectConfig or dict
            async with client.aio.live.connect(model=MODEL, config=self.config) as session:
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                async with asyncio.TaskGroup() as tg:
                    send_text_task = tg.create_task(self.send_text())
                    tg.create_task(self.send_realtime())

                    if "audio" in self.inputs:
                        tg.create_task(self.listen_audio())
                    if "camera" in self.inputs:
                        tg.create_task(self.get_frames())
                    if "screen" in self.inputs:
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

class FlexibleListAction(argparse.Action):
    def __init__(self, type_func=str, choices=None, *args, **kwargs):
        self.type_func = type_func
        self.choices = choices
        super().__init__(*args, **kwargs)
        
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, []) or []
        
        try:
            if isinstance(values, str):
                new_items = [self.type_func(v.strip()) 
                            for v in values.split(',') 
                            if v.strip()]
            else:
                new_items = [self.type_func(v) for v in values]
                
            # Validate choices
            if self.choices is not None:
                invalid_items = [str(item) for item in new_items if item not in self.choices]
                if invalid_items:
                    raise argparse.ArgumentError(
                        self,
                        f"invalid choice(s): {', '.join(invalid_items)}. "
                        f"(choose from {', '.join(str(c) for c in self.choices)})"
                    )
                    
            items.extend(new_items)
            setattr(namespace, self.dest, items)
            
        except ValueError as e:
            raise argparse.ArgumentError(self, f"invalid value: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", 
        action=FlexibleListAction,
        nargs='*',  # Accept any number of space-separated values
        help="List of inputs. Can be specified multiple times, comma-separated, or space-separated",
        choices=["camera", "screen", "text", "audio"],
        default=[]
    )
    parser.add_argument(
        "--output",
        type=str,
        default="text",
        help="Output mode. Note that output modes cannot be mixed, unlike inputs. Text_TTS is just text output with a system TTS post process, primarily for demo purposes.",
        choices=["text", "audio", "text_tts"],
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Whether or not to use Tools in the live stream",
    )
    parser.add_argument(
        "--audio_input",
        type=int,
        default="1",
        help="The microphone you want to use.",
        choices=available_audio_inputs,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug (right now) will print the 'inputs' being streamed",
    )

    args = parser.parse_args()

    # We only expect output to be TEXT or AUDIO -- never both (not yet supported)
    output = args.output.upper() if args.output != "text_tts" else "TEXT"

    config = types.LiveConnectConfig(
        response_modalities=[output],
        system_instruction=types.Content(
            parts=[
                types.Part(text="You are a helpful desktop assistant AI."),
                types.Part(text="When asked to draw bounding boxes, always use box2d formats for x1,y1 and x2,2."),
                types.Part(text="Do not ask the user for coordinates."),
                types.Part(text="Do not apologize for making mistakes unless you are told you made a mistake. Assuming the bounding box was drawn successfully."),
                # types.Part(text="You always respond in audio or text, but when you need to think about something, you call the print_on_screen tool with just the response you need."),
                # types.Part(text="If you are able to click on things in the screen, you can use that tool."),
                # types.Part(text="If you cannot find the appropriate tool, respond to the user as if you do not have tool use enabled at all."),
                # types.Part(text="When you are being asked to move the mouse, first predict the box2d location of the element and draw a bounding box."),
            ]
        )
    )

    tool_list_str = ""
    if args.tools:
        config.tools = [ tool ]
        tool_list_str = ", ".join([tool.name for tool in tool.function_declarations])

    print(f"Configuration: \n\t{args.tools=}\n\t{args.output=}\n\t{args.inputs=}\n\t{tool_list_str=}")
    main = StreamLoop(config=config, inputs=args.inputs, tts=args.output == "text_tts", debug=args.debug)
    
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pya.terminate()
