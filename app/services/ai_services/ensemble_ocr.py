import os
import io
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import re

# Google Cloud Vision imports
from google.cloud import vision
from PIL import Image, ImageDraw

# Google Generative AI imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class OCREngine(Enum):
    """Enum to represent different OCR engines"""
    VISION = "vision"
    GEMINI = "gemini"
    ENSEMBLE = "ensemble"

class MedicalDocumentType(Enum):
    """Enum for different types of medical documents"""
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    CLINICAL_NOTE = "clinical_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    INSURANCE_FORM = "insurance_form"
    UNKNOWN = "unknown"

class MedicalOCRService:
    """
    Enhanced OCR service specifically optimized for medical documents.
    Uses an ensemble approach combining traditional OCR with LLM capabilities.
    """
    
    def __init__(self, vision_credentials_path: Optional[str] = None, gemini_api_key: Optional[str] = None):
        """
        Initialize the OCR service with credentials for both Vision API and Gemini.
        
        Args:
            vision_credentials_path: Path to Google Cloud credentials JSON file for Vision API
            gemini_api_key: API key for Google Gemini
        """
        # Initialize Vision API client
        self.vision_client = vision.ImageAnnotatorClient()
        
        # Initialize Gemini
        genai.configure(api_key="AIzaSyCS2n6v5cc7exHN2tNAaiZ4vFVByl8ENnM")
        self.gemini_model = genai.GenerativeModel(model_name="gemini-2.5-pro-exp-03-25")
        
        # Safety settings for Gemini
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Common medical terms/abbreviations and their canonical forms
        self.medical_terms = {
            "rx": "prescription",
            "sig": "instructions",
            "po": "by mouth",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "prn": "as needed",
            "q.d.": "daily",
            "b.i.d.": "twice daily",
            "t.i.d.": "three times daily",
            "q.i.d.": "four times daily",
            "ml": "milliliter",
            "mg": "milligram",
            "mcg": "microgram",
            "wbc": "white blood cell count",
            "rbc": "red blood cell count",
            "hct": "hematocrit",
            "hgb": "hemoglobin",
            # Add more medical terms as needed
        }
    
    def extract_text_vision(self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> str:
        """
        Extract text using Google Cloud Vision API.
        
        Args:
            image_path: Path to the image file (optional)
            image_bytes: Raw bytes of the image (optional)
            
        Returns:
            The extracted text as a string
        """
        image = self._get_vision_image(image_path, image_bytes)
        
        # Perform document text detection
        response = self.vision_client.document_text_detection(image=image)
        
        # Check for errors
        if response.error.message:
            raise Exception(f"Error detecting document text: {response.error.message}")
        
        # Get the full text
        return response.full_text_annotation.text
    
    def extract_structured_text_vision(self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> Dict:
        """
        Extract structured text with Vision API including spatial information.
        
        Args:
            image_path: Path to the image file (optional)
            image_bytes: Raw bytes of the image (optional)
            
        Returns:
            Dictionary with structured text information
        """
        image = self._get_vision_image(image_path, image_bytes)
        
        response = self.vision_client.document_text_detection(image=image)
        document = response.full_text_annotation
        
        # Create a structured representation
        result = {
            "full_text": document.text,
            "blocks": []
        }
        
        for page in document.pages:
            for block_idx, block in enumerate(page.blocks):
                block_info = {
                    "id": block_idx,
                    "text": "",
                    "paragraphs": [],
                    "bounding_box": self._format_bounding_box(block.bounding_box)
                }
                
                for paragraph in block.paragraphs:
                    para_info = {
                        "text": "",
                        "words": [],
                        "bounding_box": self._format_bounding_box(paragraph.bounding_box)
                    }
                    
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        word_info = {
                            "text": word_text,
                            "bounding_box": self._format_bounding_box(word.bounding_box),
                            "confidence": word.confidence
                        }
                        para_info["words"].append(word_info)
                        
                    para_info["text"] = ' '.join([w["text"] for w in para_info["words"]])
                    block_info["paragraphs"].append(para_info)
                    block_info["text"] += para_info["text"] + " "
                
                result["blocks"].append(block_info)
        
        return result
    
    def _format_bounding_box(self, bounding_box) -> List[Dict[str, int]]:
        """Convert a Vision API bounding box to a standard format"""
        vertices = []
        for vertex in bounding_box.vertices:
            vertices.append({"x": vertex.x, "y": vertex.y})
        return vertices
    
    def _prep_image_for_gemini(self, image_path: str) -> Any:
        """
        Prepare an image file for use with Gemini API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            File object ready for Gemini
        """
        sample_file = genai.upload_file(path=image_path, display_name="MedicalDocument")
        return sample_file
    
    def extract_text_gemini(self, image_path: str, prompt: str = "Analyze this medical document and extract all text accurately.") -> str:
        """
        Extract text using Google Gemini generative AI.
        
        Args:
            image_path: Path to the image file
            prompt: Customized prompt for Gemini to better guide extraction
            
        Returns:
            The extracted text as a string
        """
        sample_file = self._prep_image_for_gemini(image_path)
        
        # Generate content with Gemini
        response = self.gemini_model.generate_content(
            [sample_file, prompt],
            safety_settings=self.safety_settings
        )
        
        return response.text
    
    def extract_structured_data_gemini(self, image_path: str, document_type: MedicalDocumentType = MedicalDocumentType.UNKNOWN) -> Dict:
        """
        Extract structured data from a medical document using Gemini.
        
        Args:
            image_path: Path to the image file
            document_type: Type of medical document for optimized prompting
            
        Returns:
            Dictionary with structured medical information
        """
        sample_file = self._prep_image_for_gemini(image_path)
        
        # Create a targeted prompt based on document type
        prompt = self._get_prompt_for_document_type(document_type)
        
        # Generate content with Gemini
        response = self.gemini_model.generate_content(
            [sample_file, prompt],
            safety_settings=self.safety_settings
        )
        
        # Try to parse the response as JSON
        try:
            # Check if response is in markdown code block format
            text = response.text
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            
            if json_match:
                json_text = json_match.group(1)
                data = json.loads(json_text)
            else:
                # Try to load directly if it's plain JSON
                data = json.loads(text)
                
            return data
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {"raw_text": response.text}
    
    def _get_prompt_for_document_type(self, document_type: MedicalDocumentType) -> str:
        """
        Get an optimized prompt for a specific type of medical document.
        
        Args:
            document_type: Type of medical document
            
        Returns:
            Customized prompt string
        """
        base_prompt = "Analyze this medical document carefully and extract all information in a structured format. Return the result as valid JSON."
        
        if document_type == MedicalDocumentType.PRESCRIPTION:
            return base_prompt + " Include fields for patient name, doctor name, medication names, dosages, instructions, date, pharmacy information, and any other relevant details. Format dosages consistently and standardize medical abbreviations."
        
        elif document_type == MedicalDocumentType.LAB_REPORT:
            return base_prompt + " Include fields for patient information, test names, results, reference ranges, units, collection date, report date, ordering physician, and any flags or notes. Ensure numeric values are properly formatted."
        
        elif document_type == MedicalDocumentType.CLINICAL_NOTE:
            return base_prompt + " Extract sections for chief complaint, history of present illness, past medical history, medications, allergies, review of systems, physical examination, assessment, and plan. Preserve the hierarchical structure of the note."
        
        elif document_type == MedicalDocumentType.DISCHARGE_SUMMARY:
            return base_prompt + " Include fields for patient details, admission date, discharge date, admitting diagnosis, discharge diagnosis, procedures performed, discharge medications, follow-up instructions, and any other critical information."
        
        elif document_type == MedicalDocumentType.INSURANCE_FORM:
            return base_prompt + " Extract all fields including policy numbers, group numbers, subscriber information, provider details, service dates, procedure codes, diagnosis codes, and charges."
        
        else:
            return base_prompt + " Identify what type of medical document this is and structure the data accordingly. Include all relevant fields with their values."
    
    def ensemble_ocr(self, 
                    image_path: str, 
                    document_type: MedicalDocumentType = MedicalDocumentType.UNKNOWN,
                    confidence_threshold: float = 0.85) -> Dict:
        """
        Perform ensemble OCR using both Vision API and Gemini.
        
        Args:
            image_path: Path to the image file
            document_type: Type of medical document
            confidence_threshold: Threshold for confidence in Vision API results
            
        Returns:
            Dictionary with enhanced and verified text extraction
        """
        # Step 1: Get raw structured text from Vision API
        vision_structured = self.extract_structured_text_vision(image_path=image_path)
        
        # Step 2: Get structured data from Gemini
        gemini_result = self.extract_structured_data_gemini(image_path, document_type)
        
        # Step 3: Merge results with verification and resolution
        result = self._merge_ocr_results(vision_structured, gemini_result, confidence_threshold)
        
        # Step 4: Post-process for medical context
        result = self._post_process_medical_text(result)
        
        return result
    
    def _merge_ocr_results(self, vision_result: Dict, gemini_result: Dict, confidence_threshold: float) -> Dict:
        """
        Merge OCR results from Vision API and Gemini.
        
        Args:
            vision_result: Structured text from Vision API
            gemini_result: Structured data from Gemini
            confidence_threshold: Threshold for confidence in Vision API results
            
        Returns:
            Merged and verified dictionary
        """
        merged_result = {
            "text": vision_result.get("full_text", ""),
            "structured_data": {},
            "confidence": {
                "overall": 0.0,
                "blocks": []
            },
            "verification": {
                "mismatches": [],
                "corrections": []
            }
        }
        
        # Copy over Gemini's structured interpretation if available
        if isinstance(gemini_result, dict) and "raw_text" not in gemini_result:
            merged_result["structured_data"] = gemini_result
        
        # Process blocks for verification
        low_confidence_blocks = []
        overall_confidence_sum = 0
        block_count = 0
        
        for block in vision_result.get("blocks", []):
            block_confidence = 0
            word_count = 0
            
            # Calculate average confidence for this block
            for para in block.get("paragraphs", []):
                for word_info in para.get("words", []):
                    block_confidence += word_info.get("confidence", 0)
                    word_count += 1
            
            if word_count > 0:
                avg_block_confidence = block_confidence / word_count
                
                # Track blocks with low confidence
                if avg_block_confidence < confidence_threshold:
                    low_confidence_blocks.append({
                        "block_id": block.get("id"),
                        "text": block.get("text", "").strip(),
                        "confidence": avg_block_confidence
                    })
                
                merged_result["confidence"]["blocks"].append({
                    "block_id": block.get("id"),
                    "confidence": avg_block_confidence
                })
                
                overall_confidence_sum += avg_block_confidence
                block_count += 1
        
        # Calculate overall confidence
        if block_count > 0:
            merged_result["confidence"]["overall"] = overall_confidence_sum / block_count
        
        # Attempt to resolve low confidence areas using Gemini's interpretation
        if "raw_text" not in gemini_result and low_confidence_blocks:
            for low_conf_block in low_confidence_blocks:
                block_text = low_conf_block["text"]
                
                # Check if this text appears in any Gemini structured field
                gemini_correction = self._find_correction_in_gemini(block_text, gemini_result)
                
                if gemini_correction:
                    merged_result["verification"]["corrections"].append({
                        "block_id": low_conf_block["block_id"],
                        "original": block_text,
                        "corrected": gemini_correction,
                        "confidence": low_conf_block["confidence"]
                    })
        
        return merged_result
    
    def _find_correction_in_gemini(self, text: str, gemini_result: Dict) -> Optional[str]:
        """
        Find a possible correction for low-confidence text in Gemini results.
        Uses fuzzy matching to identify similar text.
        
        Args:
            text: The text to find a correction for
            gemini_result: Structured data from Gemini
            
        Returns:
            Corrected text if found, None otherwise
        """
        # Simple implementation - in a production system, 
        # you would use a more sophisticated fuzzy matching algorithm
        
        # Function to recursively search through a dictionary
        def search_dict(d, text):
            for key, value in d.items():
                if isinstance(value, str) and self._similar_enough(text, value):
                    return value
                elif isinstance(value, dict):
                    result = search_dict(value, text)
                    if result:
                        return result
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            result = search_dict(item, text)
                            if result:
                                return result
            return None
        
        return search_dict(gemini_result, text)
    
    def _similar_enough(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are similar enough for correction purposes.
        A very basic implementation of text similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if texts are similar enough, False otherwise
        """
        # Normalize both texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # If one is significantly longer than the other, they're probably not the same
        if len(t1) > 2 * len(t2) or len(t2) > 2 * len(t1):
            return False
        
        # Check if one is a substring of the other
        if t1 in t2 or t2 in t1:
            return True
        
        # Simple character-based similarity
        # Count matching characters
        chars1 = set(t1)
        chars2 = set(t2)
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        # Jaccard similarity
        if union > 0:
            similarity = intersection / union
            return similarity > 0.6  # Arbitrary threshold
        
        return False
    
    def _post_process_medical_text(self, result: Dict) -> Dict:
        """
        Post-process OCR results in medical context.
        Standardizes medical terms, formats numbers, etc.
        
        Args:
            result: The merged OCR result
            
        Returns:
            Post-processed result
        """
        # Add post-processing functionality
        if "text" in result:
            # Standardize medical abbreviations
            processed_text = result["text"]
            
            # Replace common abbreviations with full forms
            for abbr, full_form in self.medical_terms.items():
                # Use word boundary regex to avoid partial replacements
                pattern = r'\b' + re.escape(abbr) + r'\b'
                processed_text = re.sub(pattern, full_form, processed_text, flags=re.IGNORECASE)
            
            # Standardize dosage formats
            # For example: convert "5 mg" to "5 milligrams" for consistency
            processed_text = re.sub(r'(\d+)\s*mg\b', r'\1 milligrams', processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(r'(\d+)\s*ml\b', r'\1 milliliters', processed_text, flags=re.IGNORECASE)
            
            result["processed_text"] = processed_text
        
        # If we have structured data, process it as well
        if "structured_data" in result and isinstance(result["structured_data"], dict):
            self._standardize_structured_data(result["structured_data"])
        
        return result
    
    def _standardize_structured_data(self, data: Dict) -> None:
        """
        Recursively standardize medical terms in structured data.
        Modifies the dictionary in place.
        
        Args:
            data: Dictionary to standardize
        """
        for key, value in data.items():
            if isinstance(value, str):
                # Standardize string values
                for abbr, full_form in self.medical_terms.items():
                    pattern = r'\b' + re.escape(abbr) + r'\b'
                    value = re.sub(pattern, full_form, value, flags=re.IGNORECASE)
                data[key] = value
            elif isinstance(value, dict):
                # Recurse for nested dictionaries
                self._standardize_structured_data(value)
            elif isinstance(value, list):
                # Process lists of dictionaries
                for item in value:
                    if isinstance(item, dict):
                        self._standardize_structured_data(item)
    
    def _get_vision_image(self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> vision.Image:
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
    
    def detect_document_type(self, image_path: str) -> MedicalDocumentType:
        """
        Detect the type of medical document using Gemini.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Detected document type
        """
        sample_file = self._prep_image_for_gemini(image_path)
        prompt = ("What type of medical document is this? Choose from: prescription, lab_report, " 
                 "clinical_note, discharge_summary, insurance_form, or unknown. Return only the type name.")
        
        # Generate content with Gemini
        response = self.gemini_model.generate_content(
            [sample_file, prompt],
            safety_settings=self.safety_settings
        )
        
        # Parse response to get document type
        doc_type_text = response.text.strip().lower()
        
        # Map to enum
        for doc_type in MedicalDocumentType:
            if doc_type.value in doc_type_text:
                return doc_type
        
        return MedicalDocumentType.UNKNOWN
    
    def visualize_ocr_results(self, 
                             image_path: str, 
                             output_path: Optional[str] = None,
                             show_blocks: bool = True,
                             show_confidence: bool = True,
                             highlight_corrections: bool = True) -> None:
        """
        Create a visualization of OCR results with confidence levels and corrections.
        
        Args:
            image_path: Path to the image file
            output_path: Path to save the visualization
            show_blocks: Whether to show text blocks
            show_confidence: Whether to show confidence levels
            highlight_corrections: Whether to highlight corrections
        """
        # Get ensemble results first
        result = self.ensemble_ocr(image_path)
        
        # Get structured Vision API results for visualization
        vision_result = self.extract_structured_text_vision(image_path=image_path)
        
        # Open the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Map block IDs to confidence for easy lookup
        confidence_map = {}
        for block_conf in result["confidence"].get("blocks", []):
            confidence_map[block_conf.get("block_id")] = block_conf.get("confidence", 0)
        
        # Map block IDs to corrections
        correction_map = {}
        for correction in result["verification"].get("corrections", []):
            correction_map[correction.get("block_id")] = correction
        
        # Draw blocks
        for block in vision_result.get("blocks", []):
            block_id = block.get("id")
            confidence = confidence_map.get(block_id, 0)
            
            # Determine color based on confidence
            if block_id in correction_map:
                color = "red"  # Corrected blocks
            elif confidence < 0.7:
                color = "orange"  # Low confidence
            elif confidence < 0.9:
                color = "yellow"  # Medium confidence
            else:
                color = "green"  # High confidence
            
            # Draw bounding box
            if show_blocks:
                vertices = []
                for vertex in block.get("bounding_box", []):
                    vertices.extend([vertex.get("x", 0), vertex.get("y", 0)])
                
                if vertices:
                    draw.polygon(vertices, outline=color)
            
            # Add confidence text
            if show_confidence and "bounding_box" in block and block["bounding_box"]:
                x = block["bounding_box"][0].get("x", 0)
                y = block["bounding_box"][0].get("y", 0) - 15
                confidence_text = f"{confidence:.2f}"
                draw.text((x, y), confidence_text, fill=color)
        
        # Display or save
        if output_path:
            image.save(output_path)
        else:
            image.show()

    def extract_medical_entities(self, 
                               image_path: str, 
                               document_type: Optional[MedicalDocumentType] = None) -> Dict:
        """
        Extract medical entities from a document.
        
        Args:
            image_path: Path to the image file
            document_type: Type of medical document (if known)
            
        Returns:
            Dictionary with extracted medical entities
        """
        # Auto-detect document type if not provided
        if document_type is None:
            document_type = self.detect_document_type(image_path)
        
        # Get ensemble OCR results
        ocr_result = self.ensemble_ocr(image_path, document_type)
        
        # Extract entities based on document type
        if document_type == MedicalDocumentType.PRESCRIPTION:
            return self._extract_prescription_entities(ocr_result)
        elif document_type == MedicalDocumentType.LAB_REPORT:
            return self._extract_lab_report_entities(ocr_result)
        elif document_type == MedicalDocumentType.CLINICAL_NOTE:
            return self._extract_clinical_note_entities(ocr_result)
        else:
            # For other types, use Gemini's structured data directly
            if "structured_data" in ocr_result and ocr_result["structured_data"]:
                return ocr_result["structured_data"]
            else:
                # As a fallback, ask Gemini to extract entities from the text
                return self._extract_entities_with_gemini(image_path, ocr_result.get("text", ""))
    
    def _extract_prescription_entities(self, ocr_result: Dict) -> Dict:
        """Extract entities specific to prescriptions"""
        entities = {}
        
        # Try to get from structured data first
        if "structured_data" in ocr_result and ocr_result["structured_data"]:
            return ocr_result["structured_data"]
        
        # Extract from text as fallback
        text = ocr_result.get("processed_text", ocr_result.get("text", ""))
        
        # Extract patterns for prescriptions
        # Patient name pattern (assuming format like "Patient: John Doe")
        patient_match = re.search(r'Patient(?:\s*:)?\s*([A-Za-z\s]+)', text)
        if patient_match:
            entities["patient_name"] = patient_match.group(1).strip()
        
        # Medication pattern (assuming format like "Rx: Medication Name 50mg")
        med_matches = re.finditer(r'(?:Rx|Medication)(?:\s*:)?\s*([A-Za-z\s]+)(?:\s+(\d+\s*(?:mg|mcg|ml|milligram|milligrams|microgram|micrograms)))?', text)
        
        medications = []
        for match in med_matches:
            med = {
                "name": match.group(1).strip()
            }
            if match.group(2):  # If dosage was captured
                med["dosage"] = match.group(2).strip()
            medications.append(med)
        
        if medications:
            entities["medications"] = medications
        
        # More extraction logic would be added here
        
        return entities
    
    def _extract_lab_report_entities(self, ocr_result: Dict) -> Dict:
        """Extract entities specific to lab reports"""
        # Similar implementation to prescription extraction
        # Would include patterns for lab test names, values, reference ranges, etc.
        
        # For now, just return the structured data if available
        if "structured_data" in ocr_result and ocr_result["structured_data"]:
            return ocr_result["structured_data"]
        return {"text": ocr_result.get("text", "")}
    
    def _extract_clinical_note_entities(self, ocr_result: Dict) -> Dict:
        """Extract entities specific to clinical notes"""
        # Similar implementation
        # Would include patterns for sections like HPI, PMH, etc.
        
        # For now, just return the structured data if available
        if "structured_data" in ocr_result and ocr_result["structured_data"]:
            return ocr_result["structured_data"]
        return {"text": ocr_result.get("text", "")}
    
    def _extract_entities_with_gemini(self, image_path: str, text: str) -> Dict:
        """Extract entities using Gemini as a fallback"""
        sample_file = self._prep_image_for_gemini(image_path)
        
        prompt = (f"From this medical document, extract all relevant medical entities and return them in JSON format. "
                 f"Here is the OCR text that was already extracted, but you can also look at the image directly: {text}")
        
        # Generate content with Gemini
        response = self.gemini_model.generate_content(
            [sample_file, prompt],
            safety_settings=self.safety_settings
        )
        
        # Try to parse the response as JSON
        try:
            # Check if response is in markdown code block format
            text = response.text
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            
            if json_match:
                json_text = json_match.group(1)
                data = json.loads(json_text)
            else:
                # Try to load directly if it's plain JSON
                data = json.loads(text)
                
            return data
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {"raw_text": response.text}


# Example usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Document OCR Ensemble")
    parser.add_argument("image_file", help="The image file to extract text from")
    parser.add_argument("-o", "--output", help="Output file to save the extracted text (optional)")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize OCR results")
    parser.add_argument("-e", "--engine", choices=["vision", "gemini", "ensemble"], default="ensemble",
                     help="OCR engine to use")
    parser.add_argument("-t", "--doc_type", choices=[t.value for t in MedicalDocumentType], 
                     help="Type of medical document")
    parser.add_argument("-s", "--structured", action="store_true", help="Return structured data")
    parser.add_argument("-m", "--entities", action="store_true", help="Extract medical entities")

    
    args = parser.parse_args()
    
    # Create OCR service
    ocr_service = MedicalOCRService()
    
    # Determine document type if specified
    doc_type = None
    if args.doc_type:
        for t in MedicalDocumentType:
            if t.value == args.doc_type:
                doc_type = t
                break
    else:
        # Auto-detect document type
        try:
            doc_type = ocr_service.detect_document_type(args.image_file)
            print(f"Detected document type: {doc_type.value}")
        except Exception as e:
            print(f"Could not detect document type: {e}")
            doc_type = MedicalDocumentType.UNKNOWN
    
    # Process based on selected engine
    if args.engine == "vision":
        if args.structured:
            result = ocr_service.extract_structured_text_vision(image_path=args.image_file)
            output = json.dumps(result, indent=2)
        else:
            output = ocr_service.extract_text_vision(image_path=args.image_file)
    
    elif args.engine == "gemini":
        if args.structured:
            result = ocr_service.extract_structured_data_gemini(args.image_file, doc_type)
            output = json.dumps(result, indent=2)
        else:
            output = ocr_service.extract_text_gemini(args.image_file)
    
    else:  # ensemble
        if args.entities:
            result = ocr_service.extract_medical_entities(args.image_file, doc_type)
            output = json.dumps(result, indent=2)
        elif args.structured:
            result = ocr_service.ensemble_ocr(args.image_file, doc_type)
            output = json.dumps(result, indent=2)
        else:
            result = ocr_service.ensemble_ocr(args.image_file, doc_type)
            output = result.get("processed_text", result.get("text", ""))
    
    # Output the results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    else:
        print("\nExtracted Content:")
        print("-" * 20)
        print(output)
    
    # Visualize if requested
    if args.visualize:
        viz_output = None
        if args.output:
            viz_output = f"{os.path.splitext(args.output)[0]}_viz.jpg"
        ocr_service.visualize_ocr_results(args.image_file, viz_output)
        if viz_output:
            print(f"Visualization saved to {viz_output}")


if __name__ == "__main__":
    main()