# from google import genai
# from google.genai import types
import os
import re
import traceback
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from anthropic import AnthropicVertex
from prompts.agents_prompts import (GENERAL_PHYSICIAN_PROMPT,
                                    CARDIOLOGIST_PROMPT,
                                    PATHOLOGIST_PROMPT,
                                    NEUROLOGIST_PROMPT,
                                    RADIOLOGIST_PROMPT,
                                    DERMATOLOGIST_PROMPT,
                                    ONCOLOGIST_PROMPT,
                                    PEDIATRICIAN_PROMPT,
                                    PSYCHIATRIST_PROMPT,
                                    ORTHOPEDIST_PROMPT)
import uuid
from together import Together
from app.core.config import LLM_PROJECT_ID,GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
from helpers import query_answer
PROJECT_ID = LLM_PROJECT_ID
LOCATION = "us-central1" 

def ai_call(system_prompt, u_prompt, call = "gemini"):
    if call == "anthropic":
        try:
            Anthropic_client = AnthropicVertex(project_id=os.environ["PROJECT_ID"],
                region="us-east5")
            response = Anthropic_client.messages.create(
                        system=system_prompt,
                        messages=[{"role": "user", "content": u_prompt}],
                        model="claude-3-5-sonnet@20240620",
                        max_tokens=8192,
                    )

            res = response.content[0].text
            return res
        except: 
            traceback.print_exc()
            return "Failed Claude call"

    elif call == "gemini":
        try:
            gemini_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=system_prompt)  
            response = gemini_model.generate_content(
                u_prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8000
                )
            )
            return response.text
        except Exception as e:
            traceback.print_exc()
            return f"Failed Gemini call: {str(e)}"
    else: 

        client = Together()

        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": "What are some fun things to do in New York?"}]
        )
        return response.choices[0].message.content
    
def general_physician_consultation(patient_data, qa_responses,doc_summary=None, information=None):
    """
    Primary consultation with general physician who determines if a specialist is needed.
    
    Args:
        patient_data (dict): Patient's medical history and data
        qa_responses (list): List of 10 Q&A responses from patient
        
    Returns:
        dict: Consultation results and specialist referral if needed
    """
    # Combine patient info for LLM
    patient_context = f"""
    Patient Medical Data:
    {patient_data}
    
    Patient Q&A Responses:
    {qa_responses}
    """

    user_prompt = f"Please review this patient's information and determine if they need a specialist. If a specialist is needed, include their specialty in CAPS in your response. Patient information: {patient_context}"
    if information:
        physician_knowledge = f"\n\nPhysician Knowledge: {information}\n\n"
        user_prompt += physician_knowledge

    # Consult with general physician
    gp_response = ai_call(
        GENERAL_PHYSICIAN_PROMPT,
        user_prompt
    )
    
    # Check if specialist is recommended
    specialist_match = re.search(r'[A-Z]{8,}', gp_response)
    specialist_needed = specialist_match.group(0) if specialist_match else None
    
    return {
        "gp_consultation": gp_response,
        "specialist_needed": specialist_needed
    }

def specialist_consultation(specialist_type, patient_data, gp_findings):
    """
    Consultation with a specialist doctor.
    
    Args:
        specialist_type (str): Type of specialist (e.g. CARDIOLOGIST)
        patient_data (dict): Patient's medical data
        gp_findings (str): General physician's consultation findings
        
    Returns:
        str: Specialist's consultation response
    """
    # Map specialist type to prompt
    specialist_prompts = {
        "CARDIOLOGIST": CARDIOLOGIST_PROMPT,
        "PATHOLOGIST": PATHOLOGIST_PROMPT,
        "NEUROLOGIST": NEUROLOGIST_PROMPT,
        "RADIOLOGIST": RADIOLOGIST_PROMPT,
        "DERMATOLOGIST": DERMATOLOGIST_PROMPT,
        "ONCOLOGIST": ONCOLOGIST_PROMPT,
        "PEDIATRICIAN": PEDIATRICIAN_PROMPT,
        "PSYCHIATRIST": PSYCHIATRIST_PROMPT,
        "ORTHOPEDIST": ORTHOPEDIST_PROMPT
    }
    
    specialist_prompt = specialist_prompts.get(specialist_type)
    if not specialist_prompt:
        return f"Error: Unknown specialist type {specialist_type}"
        
    specialist_context = f"""
    Patient Medical Data:
    {patient_data}
    
    General Physician's Findings:
    {gp_findings}
    """
    
    return ai_call(specialist_prompt, specialist_context)

def generate_medical_summary(gp_findings, specialist_findings=None):
    """
    Generate a flashcard-style summary of all medical findings.
    
    Args:
        gp_findings (str): General physician's consultation findings
        specialist_findings (str, optional): Specialist's consultation findings
        
    Returns:
        str: Formatted flashcard summary
    """
    summary_prompt = """Create a concise, flashcard-style summary of the following medical findings. 
    Format the output with clear headers and bullet points for easy reading by medical professionals.
    Include:
    - Key Symptoms
    - Important Findings
    - Recommendations
    - Follow-up Actions
    """
    
    findings = f"""
    General Physician Findings:
    {gp_findings}
    
    Specialist Findings (if applicable):
    {specialist_findings if specialist_findings else 'No specialist consultation required'}
    """
    
    return ai_call(summary_prompt, findings)

def validate_patient_data(patient_data):
    """
    Validate patient data structure and required fields.
    
    Args:
        patient_data (dict): Patient's medical data
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    required_fields = ['medical_history', 'current_medications', 'allergies', 'age', 'gender']
    
    if not isinstance(patient_data, dict):
        return False, "Patient data must be a dictionary"
        
    missing_fields = [field for field in required_fields if field not in patient_data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
        
    return True, ""

def validate_qa_responses(qa_responses):
    """
    Validate Q&A responses format and count.
    
    Args:
        qa_responses (list): List of Q&A responses
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not isinstance(qa_responses, list):
        return False, "Q&A responses must be a list"
        
    if len(qa_responses) != 10:
        return False, "Exactly 10 Q&A responses are required"
        
    for response in qa_responses:
        if not isinstance(response, str):
            return False, "Each Q&A response must be a string"
            
    return True, ""

def format_consultation_results(results):
    """
    Format consultation results in a human-readable way.
    
    Args:
        results (dict): Raw consultation results
        
    Returns:
        str: Formatted consultation results
    """
    if "error" in results:
        return f"Error: {results['error']}"
        
    formatted_output = []
    formatted_output.append("=== GENERAL PHYSICIAN CONSULTATION ===")
    formatted_output.append(results["gp_consultation"])
    formatted_output.append("\n")
    
    if results["specialist_needed"]:
        formatted_output.append(f"=== {results['specialist_needed']} CONSULTATION ===")
        formatted_output.append(results["specialist_consultation"])
        formatted_output.append("\n")
    
    formatted_output.append("=== MEDICAL SUMMARY ===")
    formatted_output.append(results["summary"])
    
    return "\n".join(formatted_output)

def process_medical_consultation(patient_data, qa_responses,doc_summary=None, information=None):
    """
    Main function to process a complete medical consultation.
    
    Args:
        patient_data (dict): Patient's medical history and data
        qa_responses (list): List of 10 Q&A responses from patient
        
    Returns:
        dict: Complete consultation results or error message
    """
    # Validate inputs
    is_valid_patient, patient_error = validate_patient_data(patient_data)
    if not is_valid_patient:
        return {"error": f"Invalid patient data: {patient_error}"}
        
    is_valid_qa, qa_error = validate_qa_responses(qa_responses)
    if not is_valid_qa:
        return {"error": f"Invalid Q&A responses: {qa_error}"}
    
    try:
        # Step 1: General Physician Consultation
        added_data = query_answer(patient_data)
        updated_patient_data = f"""Patient Data:
        {patient_data}
        Added Data:
        {added_data}"""
        print("Updated Patient Data: ", updated_patient_data)
        gp_result = general_physician_consultation(updated_patient_data, qa_responses)
        
        # Step 2: Specialist Consultation if needed
        specialist_result = None
        if gp_result["specialist_needed"]:
            specialist_result = specialist_consultation(
                gp_result["specialist_needed"],
                patient_data,
                gp_result["gp_consultation"]
            )
        
        # Step 3: Generate Summary
        summary = generate_medical_summary(
            gp_result["gp_consultation"],
            specialist_result
        )
        
        raw_results = {
            "gp_consultation": gp_result["gp_consultation"],
            "specialist_needed": gp_result["specialist_needed"],
            "specialist_consultation": specialist_result,
            "summary": summary
        }
        
        # Format the results
        formatted_results = format_consultation_results(raw_results)
        
        return {
            "raw": raw_results,
            "formatted": formatted_results
        }
    except Exception as e:
        return {"error": f"Error during consultation: {str(e)}"}    

def run_mock_consultation():
    """
    Run a mock medical consultation with realistic test data.
    """
    # Mock patient data
    
    mock_patient_data = {
        "medical_history": """
        - Diagnosed with hypertension 5 years ago
        - Type 2 diabetes diagnosed 3 years ago
        - History of occasional migraines
        - No previous surgeries
        - Family history of heart disease
        """,
        "current_medications": """
        - Lisinopril 10mg daily for blood pressure
        - Metformin 500mg twice daily for diabetes
        - Aspirin 81mg daily
        """,
        "allergies": "No known drug allergies",
        "age": 58,
        "gender": "male",
        "vital_signs": {
            "blood_pressure": "138/88",
            "heart_rate": 72,
            "temperature": 98.6,
            "weight": 185,
            "height": 70
        }
    }
    # Mock Q&A responses

    mock_qa_responses = [
        "Q1: What are your current symptoms? A: I've been experiencing chest discomfort and shortness of breath for the past week, especially when walking up stairs.",
        "Q2: When did these symptoms start? A: About 7 days ago, they started gradually and have been getting worse.",
        "Q3: Have you had similar symptoms before? A: No, this is the first time I've experienced chest discomfort like this.",
        "Q4: Does anything make the symptoms better or worse? A: Rest helps, but physical activity makes it worse. The symptoms are worse in the morning.",
        "Q5: Are you experiencing any other symptoms? A: Yes, I've been feeling more tired than usual and sometimes feel lightheaded.",
        "Q6: Have you noticed any changes in your weight? A: No significant changes in weight recently.",
        "Q7: Are you having any trouble sleeping? A: Yes, I've been waking up at night due to the chest discomfort.",
        "Q8: Have you had any recent illnesses or infections? A: No recent illnesses or infections.",
        "Q9: Are you currently experiencing any pain? A: Yes, I have a dull ache in my chest that sometimes radiates to my left arm.",
        "Q10: Have you noticed any changes in your appetite? A: My appetite has been normal, but I've been avoiding physical activity due to the symptoms."
    ]

    print("Starting mock medical consultation...\n")
    
    # Process the consultation
    results = process_medical_consultation(mock_patient_data, mock_qa_responses)
    
    # Print results
    print("\n=== Consultation Results ===\n")
    print(results["formatted"])
    
    # Print additional details if specialist was needed
    if results["raw"]["specialist_needed"]:
        print(f"\nSpecialist Referral: {results['raw']['specialist_needed']}")
    
    return results


# # Run the mock consultation if this file is executed directly
# if __name__ == "__main__":
#     print("/*"*100,"\n",run_mock_consultation(),"\n","/*"*100)
#
#


# RAG here patient data
##RAG question answers
### RAG at final Summary
### IMAGE DATA Bounding Bound