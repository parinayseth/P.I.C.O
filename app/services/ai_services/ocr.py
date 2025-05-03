import os
import io
from google.cloud import vision
from PIL import Image, ImageDraw
from enum import Enum

class OCRService:
    """
    A service for extracting text from medical documents and images.
    Provides multiple methods for different OCR needs.
    """
    
    def __init__(self):
        """
        Initialize the OCR service with optional credentials path.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file.
                             If None, will use environment variable.
        """

        
        self.vision_client = vision.ImageAnnotatorClient()
    
    def extract_text(self, image_path=None, image_bytes=None):
        """
        Extract all text from an image or document.
        
        Args:
            image_path: Path to the image file (optional)
            image_bytes: Raw bytes of the image (optional)
            
        Returns:
            The extracted text as a string
        """
        image = self._get_image(image_path, image_bytes)
        

        response = self.vision_client.document_text_detection(image=image)
        

        if response.error.message:
            raise Exception(f"Error detecting document text: {response.error.message}")
        

        return response.full_text_annotation.text
    
    def extract_structured_text(self, image_path=None, image_bytes=None):
        """
        Extract text with structural information (blocks, paragraphs).
        
        Args:
            image_path: Path to the image file (optional)
            image_bytes: Raw bytes of the image (optional)
            
        Returns:
            Dictionary with structured text information
        """
        image = self._get_image(image_path, image_bytes)
        
        response = self.vision_client.document_text_detection(image=image)
        document = response.full_text_annotation
        

        result = {
            "full_text": document.text,
            "blocks": []
        }
        
        for page in document.pages:
            for block_idx, block in enumerate(page.blocks):
                block_info = {
                    "id": block_idx,
                    "text": "",
                    "paragraphs": []
                }
                
                for paragraph in block.paragraphs:
                    para_info = {
                        "text": "",
                        "words": []
                    }
                    
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        para_info["words"].append(word_text)
                        
                    para_info["text"] = ' '.join(para_info["words"])
                    block_info["paragraphs"].append(para_info)
                    block_info["text"] += para_info["text"] + " "
                
                result["blocks"].append(block_info)
        
        return result
    
    def extract_medical_entities(self, image_path=None, image_bytes=None):
        """
        Extract medical entities from the document.
        
        Args:
            image_path: Path to the image file (optional)
            image_bytes: Raw bytes of the image (optional)
            
        Returns:
            Dictionary with extracted medical entities
        """

        text = self.extract_text(image_path, image_bytes)
        
        # TODO:
        # For now, we're just returning text
        # In a production system, you would integrate with a medical NLP service
        # or add custom logic to identify medical entities
        
        # Possible entity types to extract in a real implementation:
        # - Patient information
        # - Dates
        # - Medication names and dosages
        # - Diagnosis codes
        # - Lab values
        # - Vital signs
        
        return {
            "raw_text": text,
            "entities": {
                "medications": [],
                "diagnoses": [],
                "patient_info": {},
                "dates": []
            }
        }
    
    def _get_image(self, image_path=None, image_bytes=None):
        """
        Helper method to create a Vision API Image object from either
        a file path or image bytes.
        
        Args:
            image_path: Path to the image file (optional)
            image_bytes: Raw bytes of the image (optional)
            
        Returns:
            Vision API Image object
        """
        if image_path and os.path.exists(image_path):
            # Read image from file
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            return vision.Image(content=content)
        
        elif image_bytes:
            # Use provided image bytes
            return vision.Image(content=image_bytes)
        
        else:
            raise ValueError("Either image_path or image_bytes must be provided")


def create_visualized_output(ocr_service, image_path, output_path=None, visualize_blocks=True, 
                           visualize_paragraphs=True, visualize_words=True):
    """
    Utility function to create a visualization of the OCR results.
    
    Args:
        ocr_service: Instance of OCRService
        image_path: Path to the image file
        output_path: Path to save the visualization (if None, will show the image)
        visualize_blocks: Whether to highlight blocks
        visualize_paragraphs: Whether to highlight paragraphs
        visualize_words: Whether to highlight words
        
    Returns:
        None
    """
    # Open the image
    image = Image.open(image_path)
    
    # Get the Vision API Image
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    vision_image = vision.Image(content=content)
    
    # Get the document annotation
    response = ocr_service.vision_client.document_text_detection(image=vision_image)
    document = response.full_text_annotation
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Draw boxes for each element
    for page in document.pages:
        if visualize_blocks:
            for block in page.blocks:
                _draw_bounding_box(draw, block.bounding_box, "blue")
        
        if visualize_paragraphs:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    _draw_bounding_box(draw, paragraph.bounding_box, "red")
        
        if visualize_words:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        _draw_bounding_box(draw, word.bounding_box, "yellow")
    
    # Display or save the visualization
    if output_path:
        image.save(output_path)
    else:
        image.show()


def _draw_bounding_box(draw, bounding_box, color):
    """
    Helper function to draw a bounding box.
    
    Args:
        draw: PIL ImageDraw object
        bounding_box: Vision API bounding box
        color: Color to draw the box
    """
    vertices = []
    for vertex in bounding_box.vertices:
        vertices.extend([vertex.x, vertex.y])
    
    draw.polygon(vertices, outline=color)



# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Medical Document OCR Service")
#     parser.add_argument("image_file", help="The image file to extract text from")
#     parser.add_argument("-o", "--output", help="Output file to save the extracted text (optional)")
#     parser.add_argument("-v", "--visualize", action="store_true", help="Visualize text regions in the image")
#     parser.add_argument("-s", "--structured", action="store_true", help="Extract structured text information")
#     parser.add_argument("-m", "--medical", action="store_true", help="Extract medical entities")
    
#     args = parser.parse_args()
    

#     ocr_service = OCRService()
    

#     if args.medical:
#         result = ocr_service.extract_medical_entities(image_path=args.image_file)
#         output = f"Raw Text:\n{result['raw_text']}\n\nExtracted Entities:\n{result['entities']}"
#     elif args.structured:
#         result = ocr_service.extract_structured_text(image_path=args.image_file)
#         output = f"Full Text:\n{result['full_text']}\n\nStructured Content:\n"
#         for block in result["blocks"]:
#             output += f"\nBlock {block['id']}:\n{block['text']}\n"
#     else:
#         output = ocr_service.extract_text(image_path=args.image_file) # Genearte text only optimal for us
    

#     if args.output:
#         with open(args.output, 'w', encoding='utf-8') as f:
#             f.write(output)
#     else:
#         print("\nExtracted Content:")
#         print("-" * 20)
#         print(output)
    

#     if args.visualize:
#         viz_output = f"{args.output}_viz.jpg" if args.output else None
#         create_visualized_output(ocr_service, args.image_file, viz_output)