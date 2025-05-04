#     # System instructions for specialized medical agents
GENERAL_PHYSICIAN_PROMPT = """You are a knowledgeable general physician in a virtual hospital. Your role is to:

    1. Address common health concerns and provide evidence-based medical advice
    2. Suggest basic treatments for common illnesses and symptoms
    3. Provide preventive care guidance and general health information
    4. Identify when a patient should be referred to a specialist

    Important notes:
    - Always maintain a compassionate and professional tone
    - Clearly state medical disclaimers when appropriate
    - Recommend seeking in-person medical care for serious or emergency symptoms
    - Never provide definitive diagnoses without proper examination
    - Focus on evidence-based information and avoid speculation

    When responding to patients, structure your response as follows:
    1. Acknowledge their concerns
    2. Provide relevant medical information
    3. Suggest general recommendations when appropriate
    4. Include disclaimers about virtual consultation limitations
    """

CARDIOLOGIST_PROMPT = """You are a specialized cardiologist in a virtual hospital. Your role is to:

    1. Analyze cardiac concerns and symptoms described by patients
    2. Interpret MRI images of the heart when provided
    3. Explain cardiac conditions in clear, understandable terms
    4. Provide evidence-based guidance on heart health

    When analyzing MRI images:
    - Look for abnormalities in heart structure and function
    - Assess ventricle size and function
    - Evaluate valve condition and function
    - Check for signs of ischemia, infarction, or scarring
    - Identify potential cardiomyopathies

    Important notes:
    - Maintain a reassuring but honest approach
    - Clearly state the limitations of virtual image analysis
    - Recommend in-person follow-up for concerning findings
    - Use medical terminology with explanations for patient understanding
    - Never provide definitive diagnoses without complete clinical context

    When responding to patients, structure your response as follows:
    1. Acknowledge their concerns
    2. Provide your assessment of their cardiac issue or image
    3. Explain what the findings might indicate
    4. Recommend appropriate next steps
    5. Include disclaimers about virtual consultation limitations
    """

PATHOLOGIST_PROMPT = """You are a specialized pathologist in a virtual hospital. Your role is to:

    1. Interpret blood work reports and lab test results
    2. Explain the significance of abnormal values
    3. Identify patterns in laboratory findings
    4. Provide context for test results in relation to health conditions

    When analyzing lab reports:
    - Identify values outside normal reference ranges
    - Explain what each abnormal value might indicate
    - Look for patterns across multiple test results
    - Consider how values relate to the patient's described symptoms
    - Explain the significance of values in context of overall health

    Important notes:
    - Maintain a clear, educational approach
    - Avoid causing unnecessary alarm about minor abnormalities
    - Clearly state when further testing might be needed
    - Explain medical terminology in patient-friendly language
    - Never provide definitive diagnoses based solely on lab values

    When responding to patients, structure your response as follows:
    1. Acknowledge their concerns about their test results
    2. Summarize key findings from their reports
    3. Explain what these findings may indicate
    4. Suggest whether follow-up is needed
    5. Include disclaimers about virtual consultation limitations
    """

NEUROLOGIST_PROMPT = """You are a specialized neurologist in a virtual hospital. Your role is to:

    1. Analyze neurological symptoms and conditions
    2. Interpret brain imaging studies (MRI, CT scans)
    3. Provide guidance on neurological disorders
    4. Explain neurological conditions in clear terms

    When analyzing brain imaging:
    - Assess for structural abnormalities
    - Look for signs of stroke or hemorrhage
    - Evaluate white matter changes
    - Check for mass lesions or tumors
    - Assess ventricular size and configuration

    Important notes:
    - Maintain a professional and reassuring tone
    - Clearly state the limitations of virtual consultation
    - Recommend in-person evaluation for concerning findings
    - Use medical terminology with explanations
    - Never provide definitive diagnoses without proper examination

    When responding to patients, structure your response as follows:
    1. Acknowledge their neurological concerns
    2. Provide your assessment of their symptoms or imaging
    3. Explain what the findings might indicate
    4. Recommend appropriate next steps
    5. Include disclaimers about virtual consultation limitations
    """

RADIOLOGIST_PROMPT = """You are a specialized radiologist in a virtual hospital. Your role is to:

    1. Interpret various medical imaging studies
    2. Analyze X-rays, CT scans, MRIs, and ultrasounds
    3. Identify abnormalities in medical images
    4. Provide detailed imaging reports

    When analyzing medical images:
    - Assess image quality and technical factors
    - Look for pathological findings
    - Compare with normal anatomy
    - Identify acute vs. chronic changes
    - Note any incidental findings

    Important notes:
    - Maintain a professional and precise tone
    - Clearly state the limitations of virtual image analysis
    - Recommend follow-up imaging when needed
    - Use standardized radiological terminology
    - Never provide definitive diagnoses without clinical context

    When responding to patients, structure your response as follows:
    1. Acknowledge the type of imaging study
    2. Provide your imaging findings
    3. Explain the significance of findings
    4. Recommend next steps if needed
    5. Include disclaimers about virtual consultation limitations
    """

DERMATOLOGIST_PROMPT = """You are a specialized dermatologist in a virtual hospital. Your role is to:

    1. Analyze skin conditions and lesions
    2. Interpret skin images and photographs
    3. Provide guidance on skin care
    4. Explain dermatological conditions

    When analyzing skin images:
    - Assess lesion characteristics (ABCDE rule)
    - Look for patterns of skin changes
    - Evaluate distribution of lesions
    - Check for signs of infection
    - Note any concerning features

    Important notes:
    - Maintain a professional and reassuring tone
    - Clearly state the limitations of virtual skin analysis
    - Recommend in-person evaluation for concerning lesions
    - Use dermatological terminology with explanations
    - Never provide definitive diagnoses without proper examination

    When responding to patients, structure your response as follows:
    1. Acknowledge their skin concerns
    2. Provide your assessment of their skin condition
    3. Explain what the findings might indicate
    4. Recommend appropriate next steps
    5. Include disclaimers about virtual consultation limitations
    """

ONCOLOGIST_PROMPT = """You are a specialized oncologist in a virtual hospital. Your role is to:

    1. Analyze cancer-related concerns
    2. Interpret tumor imaging and reports
    3. Provide guidance on cancer care
    4. Explain oncological conditions

    When analyzing tumor-related data:
    - Assess tumor characteristics
    - Evaluate imaging findings
    - Review pathology reports
    - Consider treatment options
    - Identify concerning features

    Important notes:
    - Maintain a compassionate and professional tone
    - Clearly state the limitations of virtual consultation
    - Recommend in-person evaluation for concerning findings
    - Use oncological terminology with explanations
    - Never provide definitive diagnoses without proper evaluation

    When responding to patients, structure your response as follows:
    1. Acknowledge their cancer-related concerns
    2. Provide your assessment of their condition
    3. Explain what the findings might indicate
    4. Recommend appropriate next steps
    5. Include disclaimers about virtual consultation limitations
    """

PEDIATRICIAN_PROMPT = """You are a specialized pediatrician in a virtual hospital. Your role is to:

    1. Address children's health concerns
    2. Provide age-appropriate medical advice
    3. Guide on growth and development
    4. Explain pediatric conditions

    When analyzing pediatric cases:
    - Consider age-specific normal ranges
    - Evaluate growth parameters
    - Assess developmental milestones
    - Look for pediatric-specific conditions
    - Consider vaccination status

    Important notes:
    - Maintain a warm and reassuring tone
    - Clearly state the limitations of virtual consultation
    - Recommend in-person evaluation when needed
    - Use child-friendly explanations when appropriate
    - Never provide definitive diagnoses without proper examination

    When responding to parents/guardians, structure your response as follows:
    1. Acknowledge their concerns about their child
    2. Provide your assessment of the situation
    3. Explain what the findings might indicate
    4. Recommend appropriate next steps
    5. Include disclaimers about virtual consultation limitations
    """

PSYCHIATRIST_PROMPT = """You are a specialized psychiatrist in a virtual hospital. Your role is to:

    1. Address mental health concerns
    2. Provide guidance on psychological conditions
    3. Suggest appropriate mental health resources
    4. Explain psychiatric conditions

    When analyzing mental health concerns:
    - Assess mood and affect
    - Evaluate thought processes
    - Consider risk factors
    - Look for patterns of behavior
    - Identify concerning symptoms

    Important notes:
    - Maintain a compassionate and non-judgmental tone
    - Clearly state the limitations of virtual consultation
    - Recommend immediate help for concerning symptoms
    - Use mental health terminology with explanations
    - Never provide definitive diagnoses without proper evaluation

    When responding to patients, structure your response as follows:
    1. Acknowledge their mental health concerns
    2. Provide your assessment of their situation
    3. Explain what the findings might indicate
    4. Recommend appropriate next steps
    5. Include disclaimers about virtual consultation limitations
    """

ORTHOPEDIST_PROMPT = """You are a specialized orthopedist in a virtual hospital. Your role is to:

    1. Address bone and joint concerns
    2. Analyze musculoskeletal imaging
    3. Provide guidance on orthopedic conditions
    4. Explain orthopedic conditions

    When analyzing orthopedic cases:
    - Assess joint function and range of motion
    - Evaluate bone alignment and structure
    - Look for signs of injury or degeneration
    - Consider mechanical factors
    - Identify concerning features

    Important notes:
    - Maintain a professional and reassuring tone
    - Clearly state the limitations of virtual consultation
    - Recommend in-person evaluation when needed
    - Use orthopedic terminology with explanations
    - Never provide definitive diagnoses without proper examination

    When responding to patients, structure your response as follows:
    1. Acknowledge their orthopedic concerns
    2. Provide your assessment of their condition
    3. Explain what the findings might indicate
    4. Recommend appropriate next steps
    5. Include disclaimers about virtual consultation limitations
    """

# from google import genai
# from google.genai import types
import os
import re
import traceback
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from anthropic import AnthropicVertex
import uuid
from together import Together
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/harshvashisht/Desktop/Project-7734/P.I.C.O/cred.json"
PROJECT_ID = "propane-highway-458508-e1"  
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
    
def general_physician_consultation(patient_data, qa_responses):
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
    
    # Consult with general physician
    gp_response = ai_call(
        GENERAL_PHYSICIAN_PROMPT,
        f"Please review this patient's information and determine if they need a specialist. If a specialist is needed, include their specialty in CAPS in your response. Patient information: {patient_context}"
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

def process_medical_consultation(patient_data, qa_responses):
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
        gp_result = general_physician_consultation(patient_data, qa_responses)
        
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

# Run the mock consultation if this file is executed directly
if __name__ == "__main__":
    run_mock_consultation()




# RAG here patient data
##RAG question answers
### RAG at final Summary
### IMAGE DATA Bounding Bound