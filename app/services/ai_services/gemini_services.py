import logging
import random
import time
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
import app.core.config as config
from vertexai import generative_models
from vertexai.preview import tokenization


logger = logging.getLogger(__name__)

async def gemini_call_flash_2(
    system_prompt,
    user_prompt,
    user_id=None,
    model_name="gemini-2.0-flash-001",
    max_retries: int = 3,
    user_feedback = None,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    response_schema = None,
    temp = 0.1,
):
    """
    Simplified function to call Gemini 2.0 Flash API.

    Args:
        system_prompt (str): System prompt for the AI model
        user_prompt (str): User's input prompt
        user_id (str, optional): User identifier
        model_name (str): Gemini model to use, defaults to gemini-2.0-flash-001
        max_retries (int): Maximum number of retries. Defaults to 3.

    Returns:
        [str, int]: Gemini API response text and status code
    """
    async def prepare_content(user_prompt, user_feedback):
        """Prepare content for Gemini API call."""
        content = f"User: {user_prompt} \n\n\n Keep the generated output precise and crisp."
        
        if user_feedback is not None:
            content += f"\n\nAdditional feedback: {user_feedback}"
            
        return content
    content = await prepare_content(user_prompt, user_feedback)
    for attempt in range(max_retries):
        try:
            # Initialize Vertex AI
            vertexai.init(project=config.LLM_PROJECT_ID, location="us-east5")
            
            # Create a generative model with system instruction
            model = GenerativeModel(model_name, system_instruction=system_prompt)
            # Generate content with minimal parameters
            response = model.generate_content(
                content,
                generation_config=GenerationConfig(
                    temperature=temp,
                    max_output_tokens=8100,
                    # response_mime_type="application/json",  Uncomment for Strucutred 
                    # response_schema=response_schema,
                ),
                safety_settings={
                    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: 
                        generative_models.HarmBlockThreshold.BLOCK_NONE,
                    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: 
                        generative_models.HarmBlockThreshold.BLOCK_NONE,
                    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: 
                        generative_models.HarmBlockThreshold.BLOCK_NONE,
                    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: 
                        generative_models.HarmBlockThreshold.BLOCK_NONE,
                }
            )

            
            # Return successful response
            return response.text, 200
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            # If it's the last retry, raise the exception
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed with model {model_name}")
                return f"Error with {model_name}: {str(e)}", 500
            
            # Wait before retrying
            wait_time = min(
                initial_wait * (2**attempt) + random.uniform(0, 0.1 * initial_wait),
                max_wait,
            )

            logger.error(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    
    return "Failed to get response after all retries", 500