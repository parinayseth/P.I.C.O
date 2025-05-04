import os
from google import genai
from google.genai import types
from PIL import Image, ImageColor, ImageDraw, ImageFont
import json
import random
import re

class ImageSegmentationService:
    def __init__(self, api_key):
        """Initialize the service with Google AI API key."""
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)

    def _parse_json(self, json_input: str) -> str:
        """Parse JSON from LLM response."""
        match = re.search(r"```json\n(.*?)```", json_input, re.DOTALL)
        return match.group(1) if match else json_input
    
    def _prepare_image(self, img: Image) -> Image:
        """Prepare image for LLM processing by converting to RGB."""
        if img.mode in ['RGBA', 'P']:
            img = img.convert('RGB')
        return img

    def _call_llm(self, img: Image) -> str:
        """Call the LLM model with the image."""
        system_prompt = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        Focus on identifying medical abnormalities and anatomical structures.
        Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".
        """

        prompt = "Segment this Image for abnormalities"
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro-exp-03-25",
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.5,
                    safety_settings=[types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",

                    )],
                ),
            )
            return response.text
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")

    def _draw_bounding_boxes(self, img: Image, bounding_boxes: str) -> Image:
        """Draw bounding boxes on the image."""
        width, height = img.size
        colors = list(ImageColor.colormap.keys())
        draw = ImageDraw.Draw(img)

        try:
            bounding_boxes = self._parse_json(bounding_boxes)
            boxes = json.loads(bounding_boxes)

            for box in boxes:
                color = random.choice(colors)
                
                # Calculate absolute coordinates
                abs_y1 = int(box["box_2d"][0] / 1000 * height)
                abs_x1 = int(box["box_2d"][1] / 1000 * width)
                abs_y2 = int(box["box_2d"][2] / 1000 * height)
                abs_x2 = int(box["box_2d"][3] / 1000 * width)

                # Ensure correct ordering of coordinates
                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                # Draw rectangle and label
                draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
                draw.text(
                    (abs_x1 + 8, abs_y1 + 6),
                    box["label"],
                    fill=color,
                    font=ImageFont.load_default()
                )

            return img
        except Exception as e:
            raise Exception(f"Error drawing bounding boxes: {str(e)}")

    def segment_image(self, image_path: str) -> Image:
        try:
            # Load and prepare image
            img = Image.open(image_path)
            img = self._prepare_image(img)
            
            width, height = img.size
            resized_image = img.resize((1024, int(1024 * height / width)), Image.Resampling.LANCZOS)
            resized_image = self._prepare_image(resized_image)  # Ensure RGB mode after resize

            # Get LLM response
            llm_response = self._call_llm(resized_image)

            # Draw bounding boxes
            segmented_image = self._draw_bounding_boxes(resized_image, llm_response)
            
            return segmented_image

        except Exception as e:
            raise Exception(f"Image segmentation failed: {str(e)}")