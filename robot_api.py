
from google.genai import types
from termcolor import colored
import PIL.Image
import PIL.ImageDraw
import pyautogui
import requests


go_home = types.FunctionDeclaration(
      name="go_home",
      description="Moves the robot to its home position",
)

open_gripper = types.FunctionDeclaration(
      name="open_gripper",
      description="Opens the gripper",
)
close_gripper = types.FunctionDeclaration(
      name="close_gripper",
      description="Closes the gripper",
)

tool = types.Tool(function_declarations=[go_home, open_gripper, close_gripper])
tool_list = [ tool ]
tool_list_str = ", ".join([tool.name for tool in tool.function_declarations])

class SyncFunctionRunner():
    def __init__(self, context=None):
        self.context = context
        pass

    def exec(self, function_calls):
        # print(colored("\n\tSync Function Runner: Processing " + str(function_calls), "yellow"))
        for function_call in function_calls:
            try:
                # print(f"Running {function_call.name=} with {function_call.args=}")
                method = getattr(self, function_call.name)
                result = method(**function_call.args)

                tool_response = types.LiveClientToolResponse(
                    function_responses=[
                        types.FunctionResponse(
                            name=function_call.name,
                            response=result,
                            id=function_call.id,
                        )
                    ]
                )
                yield tool_response

            except Exception as e:
                print(colored(f"\n\tFunction Runner: Error executing command: {function_call=}\n{e}", "red"))

    def go_home(self):
        print(colored(f"\n\nGOING HOME", "green"))
        return {"success": True}

    def open_gripper(self):
        print(colored(f"\n\OPENING GRIPPER", "green"))
        return {"success": True}

    def close_gripper(self):
        print(colored(f"\n\nCLOSING GRIPPER", "green"))
        return {"success": True}