import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional
import os


API_KEY = os.getenv("GOOGLE_API_KEY")

def configure_genai(api_key: str = API_KEY):
    """Configure the genai API key."""
    if not api_key:
        raise ValueError("API key for Google Generative AI is not set.")
    genai.configure(api_key=api_key)

def upload_image(image_path: str):
    """
    Uploads an image file to Google's genai service.
    Returns the uploaded file object.
    """
    sample_file = genai.upload_file(path=image_path, display_name="Diagram")
    print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
    return sample_file

def extract_text_from_image_file(uploaded_file, prompt: str) -> Optional[str]:
    """
    Uses the Gemini model to analyze an uploaded file with a prompt.
    Returns extracted text or None.
    """
    model = genai.GenerativeModel(model_name="gemini-2.5-pro-exp-03-25", system_instruction= "Analyze the given document and carefully extract the information. Include the handwritten signature and the type of image")

    response = model.generate_content(
        [uploaded_file, prompt],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    return response.text if response else None

def ocr_image_with_prompt(image_path: str, prompt: str) -> Optional[str]:
    """
    Main OCR utility function to be called externally.
    Given an image path and a prompt, returns extracted text.
    """
    configure_genai()
    uploaded_file = upload_image(image_path)
    return extract_text_from_image_file(uploaded_file, prompt)


# if __name__ == "__main__":
#     image_path = ""
#     prompt = ""
#     result = ocr_image_with_prompt(image_path, prompt)

#     if result:
#         print("Interpreted Image:")
#         print(result)
#     else:
#         print("Failed to extract text from the image.")
