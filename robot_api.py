
from google.genai import types
from termcolor import colored
import PIL.Image
import PIL.ImageDraw
import pyautogui
import requests


# status = types.FunctionDeclaration(
#       name="status",
#       description="Gets the current status of the robot",
# )
# go_home = types.FunctionDeclaration(
#       name="go_home",
#       description="Moves the robot to its home position",
# )
# open_gripper = types.FunctionDeclaration(
#       name="open_gripper",
#       description="Opens the gripper",
# )
# close_gripper = types.FunctionDeclaration(
#       name="close_gripper",
#       description="Closes the gripper",
# )

class SyncFunctionRunner():
    def __init__(self, context=None):
        self.context = context
        pass

    def exec(self, function_calls):
        # print(colored("\n\tSync Function Runner: Processing " + str(function_calls), "yellow"))
        for function_call in function_calls:
            try:
                print(f"Running {function_call.name=} with {function_call.args=}")
                method = getattr(self, function_call.name)
                result = method(**function_call.args)
                print(f"Received {result=}")

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

    def status(self):
        # call http://er:5000/status and the JSON object response as a dictionary
        response = requests.get("http://er:5000/status")
        return response.json()

    def go_home(self):
        response = requests.post("http://er:5000/move/home")
        result = response.json()
        return result

    def open_gripper(self):
        response = requests.post("http://er:5000/move/open")
        result = response.json()
        return result

    def close_gripper(self):
        response = requests.post("http://er:5000/move/close")
        result = response.json()
        return result
    
    def wave(self):
        response = requests.post("http://er:5000/move/wave")
        result = response.json()
        return result

    def custom(self, j1, j2, j3, j4, j5, j6, type):
        print(f"Custom Movement: {j1=}, {j2=}, {j3=}, {j4=}, {j5=}, {j6=} {type=}")
        """ Moves the robot to a custom location. If the type is 'angles' it uses the joint angles, if the type is 'coords' it uses the coordinate plan with j4 as pitch, j5 as yaw and j6 as roll. j1 is x, j2 is y, j3 is z."""
        response = requests.post("http://er:5000/move/custom", data= {
            "j1": j1,
            "j2": j2,
            "j3": j3,
            "j4": j4,
            "j5": j5,
            "j6": j6,
            "type": type
        })
        result = response.json()
        return result

    def shuffle(self):
        response = requests.post("http://er:5000/move/shuffle")
        result = response.json()
        return result

    def random(self):
        response = requests.post("http://er:5000/move/random")
        result = response.json()
        return result
    

def generate_function_declarations(cls):
    function_declarations = []
    for name, method in cls.__dict__.items():
        if callable(method) and not name.startswith("_") and name != "exec":
            docstring = method.__doc__
            description = docstring.split("\n")[0] if docstring else ""
            parameters = method.__annotations__
            definition = types.FunctionDeclaration(
                    name=name,
                    description=description,
            )
            if parameters and len(parameters):
                print(f"Adding {parameters=}")
                definition.parameters = types.Schema(
                    type="OBJECT",
                    properties={
                        key: types.Schema(
                            type="STRING",
                            description=parameters[key],
                        )
                        for key in parameters
                    },
                    required=list(parameters.keys()),
                )
            function_declarations.append(definition)
    return function_declarations


# tool = types.Tool(function_declarations=[status, go_home, open_gripper, close_gripper])
# tool_list = [ tool ]
# print(tool_list)
# tool_list_str = ", ".join([tool.name for tool in tool.function_declarations])
tool = types.Tool(function_declarations=generate_function_declarations(SyncFunctionRunner))
tool_list = [ tool ]
print(tool_list)
tool_list_str = ", ".join([tool.name for tool in tool.function_declarations])