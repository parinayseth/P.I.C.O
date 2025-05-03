"""
Virtual Hospital Orchestration System using LangGraph and Pydantic

This module implements a medical orchestration system that:
1. Analyzes patient queries and uploaded medical images/reports
2. Routes them to appropriate specialized medical agents
3. Returns diagnostic responses to the patient

The system uses LangGraph for workflow management and Pydantic for type validation.
"""
import os
import json
import base64
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
from anthropic import AnthropicVertex
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.core.config import LLM_PROJECT_ID,LLM_REGION,LLM_MODEL,GOOGLE_APPLICATION_CREDENTIALS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
# Initialize AnthropicVertex client
model = AnthropicVertex(
    project_id=LLM_PROJECT_ID,
    region=LLM_REGION,
)

# Memory for conversation state
memory = MemorySaver()

# Thread configuration
thread_config = {"configurable": {"thread_id": "1"}}


# ========== Pydantic Models ==========

class AgentDecision(BaseModel):
    """Output structure for the decision agent."""
    agent: str = Field(..., description="Name of the agent to handle the query")
    reasoning: str = Field(..., description="Step-by-step reasoning for selecting this agent")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")

class MediaContent(BaseModel):
    """Model for handling various types of media content."""
    type: str = Field(..., description="Type of media (image, text, etc.)")
    content: str = Field(..., description="Base64 encoded content or text content")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata about the content")

class PatientState(MessagesState):
    """State maintained across the workflow for patient interactions."""
    agent_name: Optional[str] = Field(default=None, description="Current active medical agent")
    current_input: Optional[Union[str, Dict]] = Field(default=None, description="Input to be processed")
    output: Optional[str] = Field(default=None, description="Final output to patient")
    confidence: float = Field(default=0.0, description="Confidence in the diagnosis decision")
    media_content: Optional[List[MediaContent]] = Field(default=None, description="Media files uploaded by patient")

# ========== Agent Configuration ==========

class MedicalAgentConfig:
    """Configuration settings for the virtual hospital system."""

    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85

    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes patient queries to 
    the appropriate specialized medical agent. Your job is to analyze the patient's symptoms, concerns, or uploaded 
    medical data, and determine which specialist is best suited to handle the case.

    Available medical agents:
    1. GENERAL_PHYSICIAN - For common illnesses, general health questions, basic symptoms, and preventive care advice.
    2. CARDIOLOGIST - For All heart-related concerns, chest pain, palpitations, and analysis of cardiac MRI images.
    3. PATHOLOGIST - For interpreting blood work reports, lab results, and other diagnostic test findings.
    4. NEUROLOGIST - For brain and nervous system disorders, headaches, seizures, and analysis of brain MRI/CT scans.
    5. RADIOLOGIST - For interpreting various medical imaging studies (X-rays, CT scans, MRIs, ultrasounds).
    6. DERMATOLOGIST - For skin conditions, rashes, moles, and analysis of skin lesion images.
    7. ONCOLOGIST - For cancer-related concerns, tumor analysis, and treatment guidance.
    8. PEDIATRICIAN - For children's health issues, growth concerns, and pediatric conditions.
    9. PSYCHIATRIST - For mental health concerns, mood disorders, and psychological conditions.
    10. ORTHOPEDIST - For bone and joint issues, fractures, and musculoskeletal conditions.

    Make your decision based on these guidelines:
    - For general health questions, common symptoms, basic medical advice, use the GENERAL_PHYSICIAN.
    - For heart-related concerns, chest pain, cardiac symptoms, or if MRI images of the heart are provided, use the CARDIOLOGIST.
    - For blood test results, lab reports interpretation, or concerns about specific biomarkers, use the PATHOLOGIST.
    - For brain-related issues, headaches, seizures, or brain imaging, use the NEUROLOGIST.
    - For interpretation of medical imaging studies, use the RADIOLOGIST.
    - For skin conditions, rashes, or mole analysis, use the DERMATOLOGIST.
    - For cancer-related concerns or tumor analysis, use the ONCOLOGIST.
    - For children's health issues, use the PEDIATRICIAN.
    - For mental health concerns, use the PSYCHIATRIST.
    - For bone and joint issues, use the ORTHOPEDIST.

    If there are images or reports uploaded by the patient, analyze what type of medical data they contain before making your decision.

    You must provide your answer in JSON format with the following structure:
    {
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }
    """

    # System instructions for specialized medical agents
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


# ========== Helper Functions ==========

def parse_image_data(content: str) -> str:
    """Extract and process image data from content."""
    # This would typically involve image processing logic
    # For now, we'll just return a placeholder confirmation
    return "Image data received and processed for analysis."


def parse_blood_report(content: str) -> Dict:
    """Extract and structure blood report data from content."""
    # This would typically involve text extraction and NLP
    # For now, returning placeholder structured data
    return {
        "message": "Blood report processed",
        "parameters": ["placeholder values would be extracted here"]
    }


def parse_anthropic_response(raw_response: str) -> AgentDecision:
    """Parse JSON from Anthropic response into AgentDecision."""
    # Extract JSON string from the response if needed
    try:
        # Extract JSON if it's embedded in text
        if "{" in raw_response and "}" in raw_response:
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            json_str = raw_response[json_start:json_end]
            data = json.loads(json_str)
        else:
            data = json.loads(raw_response)

        return AgentDecision(**data)
    except json.JSONDecodeError:
        # Fallback to default if parsing fails
        return AgentDecision(
            agent="GENERAL_PHYSICIAN",
            reasoning="Failed to parse decision, defaulting to general physician",
            confidence=0.5
        )


def invoke_anthropic(prompt_text: str, image_data = None) -> str:
    """Invoke the Anthropic model with a prompt and optional image data."""
    messages = [{"role": "user", "content": prompt_text}]

    if image_data:
        # Add image content to the message
        messages[0]["content"] = json.dumps([
            {
                "type": "text",
                "text": prompt_text
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",  # You may need to adjust this based on actual image type
                    "data": image_data
                }
            }
        ])

    response = model.messages.create(
        model=LLM_MODEL,
        max_tokens=8192,
        temperature=0,
        messages=messages,
        stream=False
    )
    return response.content[0].text


def extract_image_description(image_data: str) -> str:
    """
    Analyze an image and return a description of its medical content using Anthropic's vision capabilities.
    """
    try:
        # Process image data
        vision_prompt = """
        Analyze the following medical image. Describe what you see in detail, 
        focusing on any visible medical conditions, abnormalities, or notable features.
        Specifically identify if this appears to be a cardiac MRI, X-ray, blood report, 
        or other medical image type.
        """

        # Use the updated invoke_anthropic function with image data
        return invoke_anthropic(vision_prompt, image_data)
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# ========== Specialist Tool Functions ==========
def cardiologist_tool(query, image=None):
    # Simulate specialist logic (replace with real model call as needed)
    return f"[Cardiologist] Analysis for: {query} (Image: {'Yes' if image else 'No'})"

def pathologist_tool(query, report=None):
    return f"[Pathologist] Analysis for: {query} (Report: {'Yes' if report else 'No'})"

def neurologist_tool(query, image=None):
    return f"[Neurologist] Analysis for: {query} (Image: {'Yes' if image else 'No'})"

def radiologist_tool(query, image=None):
    return f"[Radiologist] Analysis for: {query} (Image: {'Yes' if image else 'No'})"

def dermatologist_tool(query, image=None):
    return f"[Dermatologist] Analysis for: {query} (Image: {'Yes' if image else 'No'})"

def oncologist_tool(query, image=None):
    return f"[Oncologist] Analysis for: {query} (Image: {'Yes' if image else 'No'})"

def pediatrician_tool(query, image=None):
    return f"[Pediatrician] Analysis for: {query} (Image: {'Yes' if image else 'No'})"

def psychiatrist_tool(query):
    return f"[Psychiatrist] Analysis for: {query}"

def orthopedist_tool(query, image=None):
    return f"[Orthopedist] Analysis for: {query} (Image: {'Yes' if image else 'No'})"

specialist_tools = [
    {
        "name": "cardiologist",
        "description": "For all heart-related concerns, chest pain, palpitations, and analysis of cardiac MRI images.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "pathologist",
        "description": "For interpreting blood work reports, lab results, and other diagnostic test findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "report": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "neurologist",
        "description": "For brain and nervous system disorders, headaches, seizures, and analysis of brain MRI/CT scans.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "radiologist",
        "description": "For interpreting various medical imaging studies (X-rays, CT scans, MRIs, ultrasounds).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "dermatologist",
        "description": "For skin conditions, rashes, moles, and analysis of skin lesion images.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "oncologist",
        "description": "For cancer-related concerns, tumor analysis, and treatment guidance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "pediatrician",
        "description": "For children's health issues, growth concerns, and pediatric conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "psychiatrist",
        "description": "For mental health concerns, mood disorders, and psychological conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "orthopedist",
        "description": "For bone and joint issues, fractures, and musculoskeletal conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "image": {"type": "string"}
            },
            "required": ["query"]
        }
    },
]

specialist_tool_map = {
    "cardiologist": cardiologist_tool,
    "pathologist": pathologist_tool,
    "neurologist": neurologist_tool,
    "radiologist": radiologist_tool,
    "dermatologist": dermatologist_tool,
    "oncologist": oncologist_tool,
    "pediatrician": pediatrician_tool,
    "psychiatrist": psychiatrist_tool,
    "orthopedist": orthopedist_tool,
}

# ========== Multi-Tool System Prompt ==========

MULTI_TOOL_SYSTEM_PROMPT = """You are a sophisticated virtual hospital system that can consult multiple 
medical specialists as needed. For complex patient cases, you should call multiple relevant specialists 
in succession to get different perspectives.

Guidelines for multi-specialist consultation:
1. First identify ALL relevant specialists needed based on the patient's symptoms and any uploaded medical data.
2. For complex cases with overlapping symptoms, consult multiple specialists (e.g., Radiologist then Cardiologist).
3. Start with specialists who interpret medical images or reports when they're provided.
4. After collecting specialist insights, synthesize them into a coherent, comprehensive response.
5. Clearly identify which parts of the assessment come from which specialist.

Remember to use medical specialists in a logical order - start with diagnostics (lab reports, imaging) 
and then move to condition-specific specialists."""

# ========== Multi-Tool Orchestration Agent ==========

def multi_tool_orchestration_agent(query: str, media_files: list = None) -> str:
    """
    Enhanced orchestration agent that can use multiple specialist tools in succession.
    """
    # Prepare image content if available
    multimodal_content = []

    # Add text as the first content item
    multimodal_content.append({
        "type": "text",
        "text": f"Patient query: {query}"
    })

    # Add image content if available
    if media_files:
        for file in media_files:
            if file.get("type") == "image":
                image_b64 = file.get("content")
                if image_b64:
                    multimodal_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    })

    # Initialize conversation with system and user message
    messages = [
        {"role": "user", "content": multimodal_content}
    ]

    # Track all tool results to synthesize later
    all_tool_results = []
    tool_use_count = 0
    max_tool_calls = 5  # Limit number of tool calls to prevent infinite loops

    # Tool calling loop
    while tool_use_count < max_tool_calls:
        response = model.messages.create(
            model=LLM_MODEL,
            system=MULTI_TOOL_SYSTEM_PROMPT,
            max_tokens=8192,
            temperature=0,
            messages=messages,
            tools=specialist_tools,
            tool_choice={"type":"auto"},
            stream=False
        )

        # Check for tool_use blocks
        tool_call_found = False

        for block in response.content:
            if block.type == "tool_use":
                tool_call_found = True
                tool_use_count += 1
                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

                # Call the appropriate tool function
                tool_func = specialist_tool_map[tool_name]

                # Execute with proper parameters
                if tool_name == "psychiatrist":
                    tool_result = tool_func(query=tool_input.get("query", ""))
                elif tool_name == "pathologist":
                    tool_result = tool_func(
                        query=tool_input.get("query", ""),
                        report=tool_input.get("report")
                    )
                else:
                    tool_result = tool_func(
                        query=tool_input.get("query", ""),
                        image=tool_input.get("image")
                    )

                # Track this tool result
                all_tool_results.append({
                    "specialist": tool_name,
                    "result": tool_result
                })

                # Add the assistant message with tool use
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_name,
                            "input": tool_input
                        }
                    ]
                })

                # Add the tool result
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_result
                        }
                    ]
                })

                break  # Process one tool at a time

        # If no more tools are called, we're done
        if not tool_call_found:
            final_response = ""
            for block in response.content:
                if block.type == "text":
                    final_response += block.text

            # If we've collected specialist insights, append a summary request
            if all_tool_results and not final_response.strip():
                # Request synthesis of specialist insights
                synthesis_prompt = f"Based on all the specialist consultations ({', '.join([t['specialist'] for t in all_tool_results])}), provide a comprehensive assessment and recommendations for the patient."

                messages.append({
                    "role": "user",
                    "content": synthesis_prompt
                })

                # Get final synthesized response
                final_response = model.messages.create(
                    model=LLM_MODEL,
                    max_tokens=8192,
                    temperature=0,
                    messages=messages,
                    stream=False
                ).content[0].text

            return final_response

    # If we reach max tool calls, force synthesis
    if all_tool_results:
        synthesis_prompt = f"You've consulted multiple specialists ({', '.join([t['specialist'] for t in all_tool_results])}). Now synthesize their insights into a comprehensive response for the patient."

        messages.append({
            "role": "user",
            "content": synthesis_prompt
        })

        final_response = model.messages.create(
            model=LLM_MODEL,
            max_tokens=8192,
            temperature=0,
            messages=messages,
            stream=False
        ).content[0].text

        return final_response

    return "Unable to complete analysis due to system limitations. Please try again or consult with a healthcare provider directly."

def analyze_mri_with_query(query: str, image_data) -> str:
    """
    Analyze an MRI image with a given query using the multi-tool orchestration agent.

    Args:
        query (str): The patient's query or description of symptoms.
        image_data (IO): The image file sent via an API as an IO buffer.

    Returns:
        str: The response from the multi-tool orchestration agent.
    """

    # Read and encode the image from the IO buffer
    img_base64 = base64.b64encode(image_data.read()).decode("utf-8")

    # Prepare the image data
    mri_example = [
        {
            "type": "image",
            "content": img_base64,
            "metadata": {
                "description": "Cardiac MRI",
                "format": "png"
            }
        }
    ]

    # Call the multi-tool orchestration agent
    response = multi_tool_orchestration_agent(query, mri_example)
    return response


# ========== Example usage ==========
# if __name__ == "__main__":
#     import os
#     img_path = os.path.join(os.path.dirname(__file__), "img.png")
#     with open(img_path, "rb") as img_file:
#         img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
#     mri_example = [
#         {
#             "type": "image",
#             "content": img_base64,
#             "metadata": {
#                 "description": "Cardiac MRI",
#                 "format": "png"
#             }
#         }
#     ]
#
#     response_with_mri = multi_tool_orchestration_agent(
#         "I've been experiencing chest pain and shortness of breath when exercising. Can you analyze this heart MRI?",
#         mri_example
#     )
#     print("Response with MRI:", response_with_mri)
