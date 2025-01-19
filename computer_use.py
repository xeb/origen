
from google.genai import types
from termcolor import colored
import PIL.Image
import PIL.ImageDraw
import pyautogui

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
tool_list = [ tool ]
tool_list_str = ", ".join([tool.name for tool in tool.function_declarations])

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
