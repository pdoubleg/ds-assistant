import base64
import os
import re

from openai import OpenAI
from pydantic_ai.toolsets import FunctionToolset

from src.tools.utils import encode_image, base64_to_image, is_url

IMAGE_DIRECTORY = "data/images"

if not os.path.exists(IMAGE_DIRECTORY):
    os.makedirs(IMAGE_DIRECTORY)


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
    file_name = re.sub(r"\.(png|jpg|jpeg|gif|bmp)$", "", file_name, flags=re.IGNORECASE)

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
        decoded_image = base64.b64decode(image_base64)
        
        # Save as PNG
        with open(f"{IMAGE_DIRECTORY}/{file_name}.png", "wb") as f:
            f.write(decoded_image)
            
        # Save as JPEG
        with open(f"{IMAGE_DIRECTORY}/{file_name}.jpeg", "wb") as f:
            f.write(decoded_image)

    return f"Success: Files have been saved as: {file_name}.png and {file_name}.jpeg"


def generate_image_with_inputs(
    image_prompt: str, input_image_paths: list[str], file_name: str
) -> str:
    """
    Generate a new image based on a prompt and input reference images using OpenAI's image generation.

    This function takes a text prompt and reference images to generate a new image. The input
    images serve as visual context or style references for the generation process.

    Args:
        image_prompt: The prompt describing the image to generate. Should be vividly descriptive.
        input_image_paths: List of paths to reference images that will influence the generation.
        file_name: The name of the file to save the generated image. Should be snake_case with no extension.

    Returns:
        The path to the generated image or success message.
    """
    # Validate image paths
    for path in input_image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

    
    # Prepare file name
    file_name = file_name.replace(" ", "_")
    # Remove any image file extension if added in error
    file_name = re.sub(r"\.(png|jpg|jpeg|gif|bmp)$", "", file_name, flags=re.IGNORECASE)

    client = OpenAI()

    # Add reference images to provide visual context
    base64_images = [encode_image(file) for file in input_image_paths]

    # Save the uploaded images to files
    image_paths = []
    for base64_img in base64_images:
        image_path = base64_to_image(base64_img)
        image_paths.append(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": image_prompt},
            ]
            + [
                {
                    "type": "input_image",
                    "image_url": {"url": path}
                    if is_url(path)
                    else f"data:image/jpeg;base64,{encode_image(path)}",
                }
                for path in image_paths
            ],
        }
    ]
    try:
        response = client.responses.create(
            model="gpt-5",
            input=messages,
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
            decoded_image = base64.b64decode(image_base64)
            # Save as PNG
            with open(f"{IMAGE_DIRECTORY}/{file_name}.png", "wb") as f:
                f.write(decoded_image)
            # Save as JPEG
            with open(f"{IMAGE_DIRECTORY}/{file_name}.jpeg", "wb") as f:
                f.write(decoded_image)
        return f"Success: Files have been saved as: {file_name}.png and {file_name}.jpeg"

    except Exception as e:
        return f"Error generating image: {str(e)}"


image_tools = FunctionToolset(
    tools=[generate_image, generate_image_with_inputs], max_retries=5
)
