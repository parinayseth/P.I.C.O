"""
Virtual Hospital Orchestration System using LangGraph and Pydantic

This module implements a medical orchestration system that:
1. Analyzes patient queries and uploaded medical images/reports
2. Routes them to appropriate specialized medical agents
3. Returns diagnostic responses to the patient

The system uses LangGraph for workflow management and Pydantic for type validation.
"""
import os

# Ensure the path to your credentials file is correct
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/yashmangalik/Documents/my_projects/P.I.C.O/credentials.json"  # Commented out for general use

import os
import json
import base64
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
from anthropic import AnthropicVertex
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import re
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Initialize AnthropicVertex client
model = AnthropicVertex(
    project_id="lumbar-poc",
    region="us-east5",
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
    2. CARDIOLOGIST - For heart-related concerns, chest pain, palpitations, and analysis of cardiac MRI images.
    3. PATHOLOGIST - For interpreting blood work reports, lab results, and other diagnostic test findings.

    Make your decision based on these guidelines:
    - For general health questions, common symptoms, basic medical advice, use the GENERAL_PHYSICIAN.
    - For heart-related concerns, chest pain, cardiac symptoms, or if MRI images of the heart are provided, use the CARDIOLOGIST.
    - For blood test results, lab reports interpretation, or concerns about specific biomarkers, use the PATHOLOGIST.

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


def invoke_anthropic(prompt_text: str) -> str:
    """Invoke the Anthropic model with a prompt."""
    response = model.messages.create(
        model="claude-3-5-sonnet@20240620",
        max_tokens=8192,
        temperature=0,
        messages=[{"role": "user", "content": prompt_text}],
        stream=False
    )
    return response.content[0].text


def extract_image_description(image_data: str) -> str:
    """
    Analyze an image and return a description of its medical content.
    This would involve calling a vision model to analyze the image.
    """
    # In a real implementation, this would call a vision model API
    # For now, we simulate the response with a placeholder

    try:
        # Process image data
        vision_prompt = f"""
        Analyze the following medical image. Describe what you see in detail, 
        focusing on any visible medical conditions, abnormalities, or notable features.
        Specifically identify if this appears to be a cardiac MRI, X-ray, blood report, 
        or other medical image type.
        """

        # Here we would actually send the image to a vision model
        # For this example, we'll return a placeholder response
        return "This is a placeholder for image analysis. In a real implementation, we would describe the medical image content here."
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# ========== Agent Implementations ==========

def create_agent_graph():
    """Create and configure the LangGraph for medical agent orchestration."""

    # Define graph state transformations
    def route_to_agent(state: PatientState) -> Dict:
        """Make decision about which medical agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        media_content = state.get("media_content", [])

        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Create context from recent conversation history (last 3 exchanges)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)
            if isinstance(msg, HumanMessage):
                recent_context += f"Patient: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Doctor: {msg.content}\n"

        # Process any media content (images, reports)
        media_descriptions = []
        for item in media_content or []:
            if item.type == "image":
                # For images, we would use a vision model to analyze content
                description = extract_image_description(item.content)
                media_descriptions.append(f"Image analysis: {description}")
            elif item.type == "text" and "blood" in item.metadata.get("description", "").lower():
                # For blood reports
                media_descriptions.append(f"Blood report content: {item.content}")

        media_context = "\n".join(media_descriptions)

        # Combine everything for the decision input
        decision_input = f"""
        {MedicalAgentConfig.DECISION_SYSTEM_PROMPT}

        Patient query: {input_text}

        Recent conversation context:
        {recent_context}

        Media uploaded by patient:
        {media_context}

        Based on this information, which medical agent should handle this case?
        """

        # Make the decision
        decision_response = invoke_anthropic(decision_input)
        decision = parse_anthropic_response(decision_response)

        print(f"Decision: {decision.agent} with confidence {decision.confidence}")

        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision.agent,
            "confidence": decision.confidence
        }

        # Route based on agent name and confidence
        if decision.confidence < MedicalAgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "GENERAL_PHYSICIAN"}  # Default to general physician if unsure

        return {"agent_state": updated_state, "next": decision.agent}

    # Define agent execution functions
    def run_general_physician(state: PatientState):
        """Handle general medical queries."""
        print(f"Selected agent: GENERAL_PHYSICIAN")

        messages = state["messages"]
        current_input = state["current_input"]
        media_content = state.get("media_content", [])

        # Prepare input for the model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Create context from recent conversation history
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges
            if isinstance(msg, HumanMessage):
                recent_context += f"Patient: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Doctor: {msg.content}\n"

        # Process any media content
        media_descriptions = []
        for item in media_content or []:
            if item.type == "image":
                description = extract_image_description(item.content)
                media_descriptions.append(f"Image provided by patient analysis: {description}")
            elif item.type == "text":
                media_descriptions.append(f"Text document provided by patient: {item.content}")

        media_context = "\n".join(media_descriptions)

        # Combine everything for the physician prompt
        physician_prompt = f"""
        {MedicalAgentConfig.GENERAL_PHYSICIAN_PROMPT}

        Patient query: {input_text}

        Recent conversation context:
        {recent_context}

        Media provided by patient:
        {media_context}

        Please provide appropriate medical guidance to the patient's query.
        """

        response = invoke_anthropic(physician_prompt)

        return {
            **state,
            "output": AIMessage(content=response),
            "agent_name": "GENERAL_PHYSICIAN"
        }

    def run_cardiologist(state: PatientState):
        """Handle cardiology queries and cardiac MRI analysis."""
        print(f"Selected agent: CARDIOLOGIST")

        messages = state["messages"]
        current_input = state["current_input"]
        media_content = state.get("media_content", [])

        # Prepare input for the model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Create context from recent conversation history
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges
            if isinstance(msg, HumanMessage):
                recent_context += f"Patient: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Doctor: {msg.content}\n"

        # Process any MRI images
        mri_descriptions = []
        for item in media_content or []:
            if item.type == "image":
                description = extract_image_description(item.content)
                mri_descriptions.append(f"Cardiac MRI analysis: {description}")

        mri_context = "\n".join(mri_descriptions)

        # Combine everything for the cardiologist prompt
        cardiologist_prompt = f"""
        {MedicalAgentConfig.CARDIOLOGIST_PROMPT}

        Patient query: {input_text}

        Recent conversation context:
        {recent_context}

        Cardiac imaging provided by patient:
        {mri_context}

        Please provide cardiac assessment and guidance based on the information provided.
        """

        response = invoke_anthropic(cardiologist_prompt)

        return {
            **state,
            "output": AIMessage(content=response),
            "agent_name": "CARDIOLOGIST"
        }

    def run_pathologist(state: PatientState):
        """Handle blood work and lab report interpretation."""
        print(f"Selected agent: PATHOLOGIST")

        messages = state["messages"]
        current_input = state["current_input"]
        media_content = state.get("media_content", [])

        # Prepare input for the model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Create context from recent conversation history
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges
            if isinstance(msg, HumanMessage):
                recent_context += f"Patient: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Doctor: {msg.content}\n"

        # Process any lab reports or blood work
        report_content = []
        for item in media_content or []:
            if item.type == "text" and "blood" in item.metadata.get("description", "").lower():
                report_content.append(f"Blood work report content: {item.content}")
            elif item.type == "image" and "report" in item.metadata.get("description", "").lower():
                description = extract_image_description(item.content)
                report_content.append(f"Lab report image analysis: {description}")

        report_context = "\n".join(report_content)

        # Combine everything for the pathologist prompt
        pathologist_prompt = f"""
        {MedicalAgentConfig.PATHOLOGIST_PROMPT}

        Patient query: {input_text}

        Recent conversation context:
        {recent_context}

        Lab reports provided by patient:
        {report_context}

        Please provide lab interpretation and guidance based on the reports.
        """

        response = invoke_anthropic(pathologist_prompt)

        return {
            **state,
            "output": AIMessage(content=response),
            "agent_name": "PATHOLOGIST"
        }

    # Create the workflow graph
    workflow = StateGraph(PatientState)

    # Add nodes for each step
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("GENERAL_PHYSICIAN", run_general_physician)
    workflow.add_node("CARDIOLOGIST", run_cardiologist)
    workflow.add_node("PATHOLOGIST", run_pathologist)

    # Define the edges (workflow connections)
    workflow.set_entry_point("route_to_agent")

    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "GENERAL_PHYSICIAN": "GENERAL_PHYSICIAN",
            "CARDIOLOGIST": "CARDIOLOGIST",
            "PATHOLOGIST": "PATHOLOGIST"
        }
    )

    # Connect agent outputs to end
    workflow.add_edge("GENERAL_PHYSICIAN", END)
    workflow.add_edge("CARDIOLOGIST", END)
    workflow.add_edge("PATHOLOGIST", END)

    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_patient_state() -> PatientState:
    """Initialize the patient state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "output": None,
        "confidence": 0.0,
        "media_content": []
    }


def process_patient_query(query: Union[str, Dict], media_files: List[Dict] = None,
                          conversation_history: List[BaseMessage] = None) -> str:
    """
    Process a patient query through the virtual hospital system.

    Args:
        query: Patient input (text string or dict with text)
        media_files: Optional list of media files (images, reports) uploaded by the patient
        conversation_history: Optional list of previous messages

    Returns:
        Response from the appropriate medical agent
    """
    # Initialize the graph
    graph = create_agent_graph()

    # Initialize state
    state = init_patient_state()

    # Add the current query
    state["current_input"] = query

    # Process any uploaded media
    if media_files:
        media_content = []
        for file in media_files:
            file_type = file.get("type", "unknown")
            content = file.get("content", "")
            metadata = file.get("metadata", {})

            media_content.append(MediaContent(
                type=file_type,
                content=content,
                metadata=metadata
            ))

        state["media_content"] = media_content

    # Convert query to HumanMessage
    if isinstance(query, str):
        query_message = HumanMessage(content=query)
    else:
        query_message = HumanMessage(content=query.get("text", ""))

    # Add to messages
    if conversation_history:
        state["messages"] = conversation_history + [query_message]
    else:
        state["messages"] = [query_message]

    # Invoke the graph
    result = graph.invoke(state, thread_config)

    # Extract the output
    output = result.get("output", AIMessage(
        content="I'm sorry, I'm unable to provide medical guidance at this time. Please consult with a healthcare professional."))

    # Return the final output
    return output.content


# ========== Image Processing Functions ==========

def process_image(image_data: str) -> Dict:
    """
    Process uploaded image data.

    Args:
        image_data: Base64 encoded image data

    Returns:
        Dictionary with processed image info
    """
    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Extract basic image information
        width, height = image.size
        format = image.format

        # Here you would typically run image analysis, but we'll skip that for now

        return {
            "status": "success",
            "image_info": {
                "width": width,
                "height": height,
                "format": format
            },
            "analysis": "Image successfully processed for medical analysis"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to process image: {str(e)}"
        }


# Example usage
if __name__ == "__main__":
    # Test the system with a sample patient query
    response = process_patient_query(
        "I've been experiencing chest pain and shortness of breath when exercising. I'm worried about my heart."
    )
    print(f"Medical Response: {response}")

    # Example with a simulated uploaded cardiac MRI
    mri_example = [
        {
            "type": "image",
            "content": "base64_encoded_image_would_be_here",
            "metadata": {
                "description": "Cardiac MRI",
                "format": "jpeg"
            }
        }
    ]

    response_with_mri = process_patient_query(
        "Can you analyze this heart MRI and tell me if there are any issues?",
        media_files=mri_example
    )
    print(f"Cardiologist Response: {response_with_mri}")

    # Example with a simulated uploaded blood report
    blood_report_example = [
        {
            "type": "text",
            "content": "CBC Results: WBC: 7.2, RBC: 4.8, Hemoglobin: 14.2, Hematocrit: 42%, Platelets: 250\nChemistry: Sodium: 140, Potassium: 4.2, Chloride: 102, CO2: 24, BUN: 15, Creatinine: 0.9, Glucose: 95",
            "metadata": {
                "description": "Blood work results from lab"
            }
        }
    ]

    response_with_bloodwork = process_patient_query(
        "Can you explain what these blood test results mean?",
        media_files=blood_report_example
    )
    print(f"Pathologist Response: {response_with_bloodwork}")