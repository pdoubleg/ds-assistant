from openai import OpenAI
import base64
import re

from pydantic_ai.toolsets import FunctionToolset
from src.tools.logging import LoggingToolset


def generate_image(image_prompt: str, file_name: str) -> str:
    """
    Generate an image based on a prompt using a diffusion model.
    
    Args:
        image_prompt: The prompt to generate the image. Should be vividly descriptive.
        file_name: The name of the file to save the image. Should be snake_case with no extension.
    
    Returns:
        The path to the generated image.
    """
    
    # prep file name
    file_name = file_name.replace(" ", "_")
    # remove any image file extension if added in error
    file_name = re.sub(r'\.(png|jpg|jpeg|gif|bmp)$', '', file_name, flags=re.IGNORECASE)
    
    client = OpenAI() 

    response = client.responses.create(
        model="gpt-5",
        input=str(image_prompt),
        tools=[{"type": "image_generation"}],
    )

    # Save the image to a file
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]
        
    if image_data:
        image_base64 = image_data[0]
        with open(f"{file_name}.png", "wb") as f:
            f.write(base64.b64decode(image_base64))
            
    return f"Success: File has been saved to: {file_name}.png"

image_tools = FunctionToolset(tools=[generate_image], max_retries=5)
