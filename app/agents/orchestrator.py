"""
Medical agent implementation module for specialized medical intelligence agents.

This module contains concrete implementations of the various medical specialized agents:
- ConversationAgent: Handles general medical conversations and chat
- RAGAgent: Retrieves and synthesizes medical knowledge
- WebSearchAgent: Processes recent medical information from the web
- ImageAnalysisAgents: Specialized medical image analysis agents

These agents are designed to be used with the medical agent orchestration system.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
from anthropic import AnthropicVertex

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all medical agents with common functionality."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None):
        """Initialize the base agent with config and client."""
        self.config = config
        self.client = client or AnthropicVertex(
            project_id=config.project_id,
            region=config.location
        )

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for context."""
        if not history:
            return ""

        context = ""
        for msg in history[-6:]:  # Last 3 exchanges (6 messages)
            role = msg.get("role", "")
            content = msg.get("content", "")
            context += f"{role.capitalize()}: {content}\n"

        return context

    def _call_model(self, system_prompt: str, user_prompt: str,
                   max_tokens: int = None, temperature: float = None) -> str:
        """Call the Anthropic model with the given prompts."""
        try:
            response = self.client.messages.create(
                system=system_prompt,
                model=self.config.model_name,
                max_tokens=max_tokens or self.config.conversation.max_tokens,
                temperature=temperature or self.config.conversation.temperature,
                messages=[{"role": "user", "content": user_prompt}],
                stream=False
            )

            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling model: {str(e)}")
            return f"I encountered an error while processing your request. Please try again."

    def process(self, input_text: str, conversation_history: List[Dict] = None) -> Dict:
        """Process the input and generate a response. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class ConversationAgent(BaseAgent):
    """Agent for handling general medical conversations and chat."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None):
        """Initialize the conversation agent."""
        super().__init__(config, client)

    def process(self, input_text: str, conversation_history: List[Dict] = None) -> Dict:
        """Process the user input and generate a conversational response."""
        logger.info(f"Conversation Agent processing: {input_text[:50]}...")

        try:
            # Format conversation history
            history_context = self._format_conversation_history(conversation_history)

            # Create the prompt
            system_prompt = """You are a trustworthy medical assistant designed to provide accurate, 
            helpful information while maintaining a conversational tone. Your answers should be clear, 
            concise, and medically sound. Remember to clarify that you're an AI without medical 
            credentials when providing health information."""

            user_prompt = f"""
            User query: {input_text}
            
            Recent conversation context: 
            {history_context}
            
            Please respond to the user in a natural, conversational manner. If they're asking
            about medical topics, be informative but include appropriate disclaimers. If they're
            engaging in general conversation, respond naturally while maintaining professionalism.
            """

            # Get response from model
            response = self._call_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.config.conversation.max_tokens,
                temperature=self.config.conversation.temperature
            )

            return {
                "response": response,
                "confidence": 0.92,
                "sources": [],
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in conversation agent: {str(e)}")
            return {
                "response": "I apologize, but I encountered an issue while processing your request. Please try again.",
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }


class RAGAgent(BaseAgent):
    """Agent for medical knowledge retrieval and synthesis."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None):
        """Initialize the RAG agent."""
        super().__init__(config, client)
        self.vector_db = None  # In production, initialize vector DB connection here

    def _retrieve_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant documents for the query."""
        # This is a simulation - in production this would connect to a vector database

        # Simulated medical knowledge base with retrieval scores
        knowledge_base = [
            {
                "content": "Diabetes mellitus, commonly known as diabetes, is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period. Symptoms often include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many health complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, damage to the nerves, damage to the eyes, and cognitive impairment.",
                "source": "Medical Encyclopedia, 2023",
                "keywords": ["diabetes", "blood sugar", "insulin", "glucose", "type 2", "type 1"]
            },
            {
                "content": "Hypertension (HTN or HT), also known as high blood pressure (HBP), is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms. Long-term high blood pressure, however, is a major risk factor for stroke, coronary artery disease, heart failure, atrial fibrillation, peripheral arterial disease, vision loss, chronic kidney disease, and dementia.",
                "source": "Journal of Hypertension, 2024",
                "keywords": ["hypertension", "blood pressure", "high blood pressure", "HBP", "HTN"]
            },
            {
                "content": "COVID-19 is a contagious disease caused by the SARS-CoV-2 virus. Most people infected with the virus experience mild to moderate respiratory illness and recover without requiring special treatment. However, some become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness.",
                "source": "WHO Coronavirus Disease Guidelines, 2024",
                "keywords": ["covid", "coronavirus", "covid-19", "sars-cov-2", "pandemic"]
            },
            {
                "content": "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or making it difficult for the bacteria to grow and multiply. Antibiotics only work against bacteria, not viruses. Antibiotics won't work for viral infections such as common cold, most sore throats, and the flu.",
                "source": "National Institutes of Health, 2023",
                "keywords": ["antibiotics", "bacteria", "infection", "antimicrobial", "resistance"]
            },
            {
                "content": "The endocrine system is a network of glands and organs that produce, store, and secrete hormones. The endocrine system influences heart rate, metabolism, appetite, mood, sexual function, reproduction, sleep cycles, and other body functions. Examples of endocrine glands include the pituitary, thyroid, parathyroid, adrenal, and pineal glands.",
                "source": "Endocrinology Journal, 2024",
                "keywords": ["endocrine", "hormone", "thyroid", "pituitary", "insulin", "diabetes"]
            },
            {
                "content": "The guidelines for treating high cholesterol focus on overall cardiovascular health. First-line treatment includes lifestyle changes such as a heart-healthy diet, regular exercise, weight management, and smoking cessation. If these measures are insufficient, medication may be prescribed, with statins being the most common. Other medications include PCSK9 inhibitors, bile acid sequestrants, and ezetimibe.",
                "source": "American Heart Association, 2024",
                "keywords": ["cholesterol", "statins", "cardiovascular", "heart", "lipids"]
            },
            {
                "content": "Asthma is a chronic condition affecting the airways in the lungs. During an asthma attack, the airways become inflamed, narrow, and produce extra mucus, making breathing difficult. Symptoms include wheezing, shortness of breath, chest tightness, and coughing. Asthma can be managed with proper treatment but cannot be cured. Treatment typically involves avoiding triggers, using rescue inhalers, and taking long-term control medications.",
                "source": "Respiratory Medicine Journal, 2023",
                "keywords": ["asthma", "breathing", "lungs", "inhaler", "wheezing"]
            }
        ]

        # Simple keyword matching for simulation
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        results = []
        for doc in knowledge_base:
            # Calculate a simple relevance score based on keyword matches
            matches = sum(1 for kw in doc["keywords"] if any(w in kw for w in query_words))
            if matches > 0:
                # Create a copy to avoid modifying the original
                result = doc.copy()
                result["relevance"] = min(0.5 + (matches * 0.1), 0.98)  # Scale between 0.5 and 0.98
                results.append(result)

        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]

    def process(self, input_text: str, conversation_history: List[Dict] = None) -> Dict:
        """Process the user input using RAG techniques."""
        logger.info(f"RAG Agent processing: {input_text[:50]}...")

        try:
            # Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(input_text, limit=self.config.rag.context_limit)

            if not retrieved_docs:
                # If no relevant documents found, return low confidence response
                return {
                    "response": "I don't have specific information about that medical topic in my knowledge base. Would you like me to provide general information or clarify your question?",
                    "confidence": 0.4,
                    "sources": [],
                    "success": True,
                    "insufficient_info": True
                }

            # Format retrieved content for the prompt
            retrieval_context = ""
            for i, doc in enumerate(retrieved_docs):
                retrieval_context += f"[{i+1}] {doc['content']} (Source: {doc['source']})\n\n"

            # Create the prompt
            system_prompt = """You are a medical knowledge assistant that provides accurate information 
            based on verified medical sources. When answering, cite your sources using the reference 
            numbers provided. Be comprehensive but concise, and acknowledge knowledge gaps rather than 
            speculating. Include appropriate medical disclaimers when necessary."""

            user_prompt = f"""
            User query: {input_text}
            
            Retrieved medical information:
            {retrieval_context}
            
            Based on the retrieved medical information, provide a comprehensive and accurate response to the 
            user's query. Cite sources appropriately using the source numbers in brackets like [1], [2], etc.
            Ensure your response is medically accurate and includes appropriate disclaimers.
            """

            # Get response from model
            response = self._call_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.config.rag.max_tokens,
                temperature=self.config.rag.temperature
            )

            # Calculate overall confidence based on retrieval relevance
            avg_confidence = sum(doc.get("relevance", 0) for doc in retrieved_docs) / len(retrieved_docs)

            return {
                "response": response,
                "confidence": avg_confidence,
                "sources": [doc["source"] for doc in retrieved_docs],
                "retrieved_documents": retrieved_docs,
                "success": True,
                "insufficient_info": avg_confidence < self.config.rag.min_retrieval_confidence
            }

        except Exception as e:
            logger.error(f"Error in RAG agent: {str(e)}")
            return {
                "response": "I apologize, but I encountered an issue while retrieving medical information. Please try again.",
                "confidence": 0.0,
                "error": str(e),
                "success": False,
                "insufficient_info": True
            }


class WebSearchAgent(BaseAgent):
    """Agent for processing recent medical information from web searches."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None):
        """Initialize the web search agent."""
        super().__init__(config, client)

    def _search_web(self, query: str, limit: int = 5) -> List[Dict]:
        """Search the web for recent medical information."""
        # This is a simulation - in production this would connect to a search API

        # Simulated search results
        search_results = [
            {
                "title": "CDC Updates COVID-19 Guidelines (May 2025)",
                "url": "https://www.cdc.gov/coronavirus/2019-ncov/your-health/index.html",
                "snippet": "The CDC has updated its COVID-19 guidelines to reflect new research on treatment protocols and prevention measures. The updated guidelines emphasize personalized risk assessment based on community transmission levels and individual health factors.",
                "date": "2025-05-01",
                "relevance": 0.95
            },
            {
                "title": "WHO Report on Emerging Infectious Diseases",
                "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019",
                "snippet": "The World Health Organization released a report on emerging infectious diseases and current treatment recommendations, including a special focus on antimicrobial resistance trends in 2025.",
                "date": "2025-04-28",
                "relevance": 0.87
            },
            {
                "title": "New Study Finds Link Between Gut Microbiome and Alzheimer's",
                "url": "https://www.nih.gov/news-events/news-releases/gut-microbiome-alzheimers-link",
                "snippet": "A groundbreaking study published in Nature Neuroscience has identified specific gut bacteria compositions that may contribute to Alzheimer's disease progression, suggesting potential for new diagnostic and therapeutic approaches.",
                "date": "2025-04-15",
                "relevance": 0.82
            },
            {
                "title": "FDA Approves Novel Diabetes Monitoring Device",
                "url": "https://www.fda.gov/news-events/press-announcements/fda-approves-novel-diabetes-monitoring",
                "snippet": "The FDA has approved a new continuous glucose monitoring device that requires no finger pricks and can be worn for up to 30 days. The device uses advanced sensor technology and machine learning to predict glucose trends.",
                "date": "2025-03-22",
                "relevance": 0.79
            },
            {
                "title": "American Heart Association Updates Hypertension Guidelines",
                "url": "https://www.heart.org/en/news/2025/04/10/hypertension-guidelines-update",
                "snippet": "The American Heart Association has released updated guidelines for hypertension management, incorporating new evidence on lifestyle interventions and medication protocols for patients with comorbidities.",
                "date": "2025-04-10",
                "relevance": 0.91
            },
            {
                "title": "Mayo Clinic Develops AI Tool for Early Cancer Detection",
                "url": "https://www.mayoclinic.org/news/ai-cancer-detection-tool",
                "snippet": "Mayo Clinic researchers have developed a new AI algorithm that can detect subtle imaging patterns indicating early-stage cancer, potentially improving survival rates through earlier intervention.",
                "date": "2025-04-05",
                "relevance": 0.76
            }
        ]

        # Filter based on query relevance (in production would use actual search results)
        filtered_results = []
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        for result in search_results:
            # Simple relevance check for the demo
            title_and_snippet = (result["title"] + " " + result["snippet"]).lower()
            matches = sum(1 for word in query_words if word in title_and_snippet)
            if matches > 0:
                # Adjust relevance based on query match
                adjusted_relevance = min(result["relevance"] * (1 + 0.05 * matches), 0.99)
                adjusted_result = result.copy()
                adjusted_result["relevance"] = adjusted_relevance
                filtered_results.append(adjusted_result)

        # Sort by relevance and limit results
        filtered_results.sort(key=lambda x: x["relevance"], reverse=True)
        return filtered_results[:limit]

    def process(self, input_text: str, conversation_history: List[Dict] = None) -> Dict:
        """Process the user input using web search."""
        logger.info(f"Web Search Agent processing: {input_text[:50]}...")

        try:
            # Search the web
            search_results = self._search_web(input_text, limit=self.config.web_search.context_limit)

            if not search_results:
                # If no relevant results found
                return {
                    "response": "I couldn't find recent information about that medical topic. Would you like me to provide general information instead?",
                    "confidence": 0.4,
                    "sources": [],
                    "success": True
                }

            # Format search results for the prompt
            search_context = ""
            for i, result in enumerate(search_results):
                search_context += f"[{i+1}] {result['title']}\n"
                search_context += f"URL: {result['url']}\n"
                search_context += f"Date: {result['date']}\n"
                search_context += f"Summary: {result['snippet']}\n\n"

            # Create the prompt
            system_prompt = """You are a medical web search assistant specializing in finding recent 
            medical information. When responding to users, synthesize the search results into a 
            coherent summary, cite your sources, and emphasize that the information is recent 
            and should be verified with healthcare professionals."""

            user_prompt = f"""
            User query: {input_text}
            
            Recent web search results:
            {search_context}
            
            Based on these recent search results from May 2025, provide a helpful summary of current information
            relevant to the user's query. Cite sources using the reference numbers provided [1], [2], etc.
            Emphasize that this information is based on recent web searches and should be verified 
            with healthcare professionals.
            """

            # Get response from model
            response = self._call_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.config.web_search.max_tokens,
                temperature=self.config.web_search.temperature
            )

            # Calculate confidence based on search result relevance
            avg_confidence = sum(result.get("relevance", 0) for result in search_results) / len(search_results)

            return {
                "response": response,
                "confidence": avg_confidence,
                "sources": [result["url"] for result in search_results],
                "search_results": search_results,
                "search_timestamp": "2025-05-02T10:00:00Z",
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in web search agent: {str(e)}")
            return {
                "response": "I apologize, but I encountered an issue while searching for recent medical information. Please try again.",
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }


class MedicalImageAgent(BaseAgent):
    """Base agent for medical image analysis."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None, image_type: str = "UNKNOWN"):
        """Initialize the medical image agent."""
        super().__init__(config, client)
        self.image_type = image_type

    def _analyze_image(self, image_data: Dict, query_text: str) -> Dict:
        """Analyze the medical image."""
        # This would connect to specialized CV models in production
        # For now, return simulated analysis based on image type

        agent_responses = {
            "BRAIN_MRI": {
                "findings": [
                    "No evidence of space-occupying lesions",
                    "Normal ventricle size and configuration",
                    "No midline shift or mass effect",
                    "No abnormal enhancement pattern"
                ],
                "impression": "Normal brain MRI with no evidence of tumor or abnormality",
                "confidence": 0.90
            },
            "CHEST_XRAY": {
                "findings": [
                    "Clear lung fields bilaterally",
                    "No evidence of consolidation or effusion",
                    "Normal cardiac silhouette",
                    "No pneumothorax or pleural effusion"
                ],
                "impression": "Normal chest radiograph with no acute cardiopulmonary process",
                "confidence": 0.88
            },
            "SKIN_LESION": {
                "findings": [
                    "Asymmetrical border",
                    "Varied coloration",
                    "Diameter approximately 7mm",
                    "Irregular borders"
                ],
                "impression": "Features concerning for possible melanoma; recommend dermatology referral for biopsy",
                "confidence": 0.85
            },
            "UNKNOWN": {
                "findings": [
                    "Image quality sufficient for basic analysis",
                    "No obvious critical abnormalities detected"
                ],
                "impression": "General medical image requires specialist interpretation; insufficient information for definitive analysis",
                "confidence": 0.60
            }
        }

        # Get response for image type, default to unknown if not found
        if self.image_type in agent_responses:
            return agent_responses[self.image_type]
        else:
            return agent_responses["UNKNOWN"]

    def process(self, input_text: str, image_data: Dict, conversation_history: List[Dict] = None) -> Dict:
        """Process the image and generate analysis."""
        logger.info(f"Medical Image Agent processing: {self.image_type}")

        try:
            # Analyze the image
            analysis = self._analyze_image(image_data, input_text)

            # Format the findings
            findings_text = "\n".join([f"- {finding}" for finding in analysis["findings"]])

            # Create the response
            response = f"""
            # Medical Image Analysis Results
            
            ## Findings:
            {findings_text}
            
            ## Impression:
            {analysis["impression"]}
            
            *Note: This is an AI analysis and should be confirmed by a qualified healthcare professional.*
            """

            return {
                "response": response,
                "confidence": analysis["confidence"],
                "findings": analysis["findings"],
                "impression": analysis["impression"],
                "image_type": self.image_type,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in medical image analysis: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an issue while analyzing the {self.image_type} image. Please try again.",
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }


class BrainMRIAgent(MedicalImageAgent):
    """Agent for analyzing brain MRI images."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None):
        """Initialize the brain MRI agent."""
        super().__init__(config, client, image_type="BRAIN_MRI")

    def _analyze_image(self, image_data: Dict, query_text: str) -> Dict:
        """Analyze brain MRI image."""
        # Call specialized brain MRI models here in production

        # Check for keywords in the query for simulation purposes
        query_lower = query_text.lower()

        if "tumor" in query_lower or "cancer" in query_lower:
            return {
                "findings": [
                    "Irregular mass identified in the right temporal lobe",
                    "Dimensions approximately 2.3 x 1.8 cm",
                    "Surrounding vasogenic edema observed",
                    "Mass demonstrates heterogeneous enhancement"
                ],
                "impression": "Findings consistent with high-grade glioma. Urgent neurosurgical consultation recommended.",
                "confidence": 0.87
            }
        elif "stroke" in query_lower or "ischemic" in query_lower:
            return {
                "findings": [
                    "Hyperintense signal on DWI in the left middle cerebral artery territory",
                    "Corresponding hypointensity on ADC map",
                    "No hemorrhagic conversion identified",
                    "No significant mass effect or midline shift"
                ],
                "impression": "Findings consistent with acute ischemic stroke in the left MCA territory. Recommend urgent stroke neurology consultation.",
                "confidence": 0.89
            }
        else:
            # Default to normal MRI findings
            return super()._analyze_image(image_data, query_text)


class ChestXRayAgent(MedicalImageAgent):
    """Agent for analyzing chest X-ray images."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None):
        """Initialize the chest X-ray agent."""
        super().__init__(config, client, image_type="CHEST_XRAY")

    def _analyze_image(self, image_data: Dict, query_text: str) -> Dict:
        """Analyze chest X-ray image."""
        # Call specialized chest X-ray models here in production

        # Check for keywords in the query for simulation purposes
        query_lower = query_text.lower()

        if "pneumonia" in query_lower or "infection" in query_lower:
            return {
                "findings": [
                    "Focal consolidation in the right lower lobe",
                    "Mild right pleural effusion",
                    "Normal cardiac silhouette",
                    "No pneumothorax identified"
                ],
                "impression": "Findings consistent with right lower lobe pneumonia. Recommend antibiotics and follow-up imaging.",
                "confidence": 0.91
            }
        elif "covid" in query_lower or "coronavirus" in query_lower:
            return {
                "findings": [
                    "Diffuse bilateral ground-glass opacities",
                    "Peripheral and basal predominance of infiltrates",
                    "No pleural effusion",
                    "Normal cardiac size"
                ],
                "impression": "Imaging pattern highly suspicious for COVID-19 pneumonia. Recommend RT-PCR testing and isolation.",
                "confidence": 0.88
            }
        else:
            # Default to normal chest X-ray findings
            return super()._analyze_image(image_data, query_text)


class SkinLesionAgent(MedicalImageAgent):
    """Agent for analyzing skin lesion images."""

    def __init__(self, config: Any, client: Optional[AnthropicVertex] = None):
        """Initialize the skin lesion agent."""
        super().__init__(config, client, image_type="SKIN_LESION")

    def _analyze_image(self, image_data: Dict, query_text: str) -> Dict:
        """Analyze skin lesion image."""
        # Call specialized dermatology models here in production

        # Check for keywords in the query for simulation purposes
        query_lower = query_text.lower()

        if "melanoma" in query_lower or "cancer" in query_lower:
            return {
                "findings": [
                    "Asymmetrical lesion with irregular borders",
                    "Varied pigmentation with dark and light areas",
                    "Diameter > 6mm (approximately 9mm)",
                    "Evidence of evolution/change reported by patient"
                ],
                "impression": "Features highly concerning for melanoma. Urgent dermatology consultation and excisional biopsy recommended.",
                "confidence": 0.89
            }
        elif "benign" in query_lower or "mole" in query_lower:
            return {
                "findings": [
                    "Symmetrical, round lesion",
                    "Regular, smooth borders",
                    "Uniform light-brown coloration",
                    "Diameter < 5mm (approximately 3mm)"
                ],
                "impression": "Features consistent with benign melanocytic nevus (common mole). Routine dermatology follow-up recommended.",
                "confidence": 0.93
            }
        else:
            # Default to concerning skin lesion findings
            return super()._analyze_image(image_data, query_text)


class AgentFactory:
    """Factory for creating and managing medical agents."""

    @staticmethod
    def create_agent(agent_type: str, config: Any, client: Optional[AnthropicVertex] = None) -> BaseAgent:
        """Create an agent instance based on the specified type."""
        agent_map = {
            "CONVERSATION_AGENT": ConversationAgent,
            "RAG_AGENT": RAGAgent,
            "WEB_SEARCH_PROCESSOR_AGENT": WebSearchAgent,
            "BRAIN_TUMOR_AGENT": BrainMRIAgent,
            "CHEST_XRAY_AGENT": ChestXRayAgent,
            "SKIN_LESION_AGENT": SkinLesionAgent
        }

        if agent_type not in agent_map:
            logger.warning(f"Unknown agent type: {agent_type}. Defaulting to conversation agent.")
            return ConversationAgent(config, client)

        return agent_map[agent_type](config, client)


# Example usage
if __name__ == "__main__":
    # Configuration object simulation
    class Config:
        def __init__(self):
            self.project_id = "lumbar-poc"
            self.location = "us-east5"
            self.model_name = "claude-3-5-sonnet@20240620"

            # Configs for different agents
            self.conversation = type('obj', (object,), {
                'max_tokens': 4096,
                'temperature': 0.7
            })

            self.rag = type('obj', (object,), {
                'max_tokens': 4096,
                'temperature': 0.2,
                'min_retrieval_confidence': 0.7,
                'context_limit': 10
            })

            self.web_search = type('obj', (object,), {
                'max_tokens': 4096,
                'temperature': 0.3,
                'context_limit': 5
            })

    # Create configuration
    config = Config()

    # Create and test conversation agent
    conversation_agent = AgentFactory.create_agent("CONVERSATION_AGENT", config)
    conv_result = conversation_agent.process(
        "What are the common symptoms of diabetes?",
        conversation_history=[
            {"role": "user", "content": "Hello, I have some medical questions."},
            {"role": "assistant", "content": "Hello! I'd be happy to help with your medical questions. What would you like to know?"}
        ]
    )
    print("\n--- Conversation Agent Result ---")
    print(f"Response: {conv_result['response']}")
    print(f"Confidence: {conv_result['confidence']}")
    print(f"Success: {conv_result['success']}")

    # Test RAG agent
    rag_agent = AgentFactory.create_agent("RAG_AGENT", config)
    rag_result = rag_agent.process("What are the complications of untreated hypertension?")
    print("\n--- RAG Agent Result ---")
    print(f"Response: {rag_result['response']}")
    print(f"Confidence: {rag_result['confidence']}")
    print(f"Sources: {rag_result['sources']}")
    print(f"Success: {rag_result['success']}")

    # Test web search agent
    web_agent = AgentFactory.create_agent("WEB_SEARCH_PROCESSOR_AGENT", config)
    web_result = web_agent.process("What are the latest COVID-19 guidelines?")
    print("\n--- Web Search Agent Result ---")
    print(f"Response: {web_result['response']}")
    print(f"Confidence: {web_result['confidence']}")
    print(f"Sources: {web_result['sources']}")
    print(f"Success: {web_result['success']}")

    # Test image analysis agents
    # In production, this would include actual image data
    mock_image_data = {"format": "JPEG", "dimensions": "1024x1024", "metadata": {"acquisition_date": "2025-04-30"}}

    # Brain MRI agent test
    brain_mri_agent = AgentFactory.create_agent("BRAIN_TUMOR_AGENT", config)
    brain_result = brain_mri_agent.process("Check for signs of tumor", mock_image_data)
    print("\n--- Brain MRI Analysis Result ---")
    print(f"Response: {brain_result['response']}")
    print(f"Confidence: {brain_result['confidence']}")
    print(f"Success: {brain_result['success']}")

    # Chest X-ray agent test
    chest_xray_agent = AgentFactory.create_agent("CHEST_XRAY_AGENT", config)
    chest_result = chest_xray_agent.process("Analyze for pneumonia", mock_image_data)
    print("\n--- Chest X-ray Analysis Result ---")
    print(f"Response: {chest_result['response']}")
    print(f"Confidence: {chest_result['confidence']}")
    print(f"Success: {chest_result['success']}")

    # Skin lesion agent test
    skin_agent = AgentFactory.create_agent("SKIN_LESION_AGENT", config)
    skin_result = skin_agent.process("Check if this mole is benign", mock_image_data)
    print("\n--- Skin Lesion Analysis Result ---")
    print(f"Response: {skin_result['response']}")
    print(f"Confidence: {skin_result['confidence']}")
    print(f"Success: {skin_result['success']}")

    # Example of handling multiple agent orchestration
    def orchestrate_agents(user_query: str, conversation_history: List[Dict] = None) -> Dict:
        """Orchestrate multiple agents based on the query type."""
        # Simple keyword-based routing for demonstration
        query_lower = user_query.lower()

        # Check for medical knowledge queries
        if any(kw in query_lower for kw in ["what is", "how does", "explain", "describe"]):
            agent = AgentFactory.create_agent("RAG_AGENT", config)
            return agent.process(user_query, conversation_history)

        # Check for recent/current information queries
        elif any(kw in query_lower for kw in ["latest", "recent", "update", "new", "current"]):
            agent = AgentFactory.create_agent("WEB_SEARCH_PROCESSOR_AGENT", config)
            return agent.process(user_query, conversation_history)

        # Default to conversation agent for general queries
        else:
            agent = AgentFactory.create_agent("CONVERSATION_AGENT", config)
            return agent.process(user_query, conversation_history)

    # Test orchestration with different query types
    queries = [
        "What is the treatment for type 2 diabetes?",
        "What are the latest guidelines for COVID-19 treatment?",
        "Can you help me understand my lab results?"
    ]

    print("\n--- Agent Orchestration Test ---")
    for query in queries:
        print(f"\nQuery: {query}")
        result = orchestrate_agents(query)
        print(f"Selected agent type: {type(result).__name__}")
        print(f"Response: {result['response'][:100]}...")  # Show first 100 chars
        print(f"Confidence: {result['confidence']}")
        print(f"Success: {result['success']}")

    # Example of how to build a complete system
    def process_medical_query(query: str, image_data: Dict = None, conversation_history: List[Dict] = None) -> Dict:
        """Complete system for processing medical queries with multi-agent orchestration."""
        try:
            # Determine if query is image-related
            if image_data:
                # Detect image type (in production, this would use CV models)
                if "brain" in query.lower() or "mri" in query.lower():
                    agent = AgentFactory.create_agent("BRAIN_TUMOR_AGENT", config)
                elif "chest" in query.lower() or "xray" in query.lower() or "x-ray" in query.lower():
                    agent = AgentFactory.create_agent("CHEST_XRAY_AGENT", config)
                elif "skin" in query.lower() or "mole" in query.lower() or "lesion" in query.lower():
                    agent = AgentFactory.create_agent("SKIN_LESION_AGENT", config)
                else:
                    # Default medical image analysis
                    return {
                        "response": "I need more information about the type of medical image you're sharing. Is this a brain MRI, chest X-ray, or skin lesion image?",
                        "confidence": 0.5,
                        "success": True,
                        "needs_clarification": True
                    }

                return agent.process(query, image_data, conversation_history)
            else:
                # Text-based query orchestration
                return orchestrate_agents(query, conversation_history)

        except Exception as e:
            logger.error(f"Error in medical query processing: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while processing your medical query. Please try again or rephrase your question.",
                "confidence": 0.0,
                "error": str(e),
                "success": False
            }

    # Test complete system
    print("\n--- Complete System Test ---")

    # Test text-only query
    text_query = "What are the warning signs of a heart attack?"
    text_result = process_medical_query(text_query)
    print(f"\nText Query: {text_query}")
    print(f"Response: {text_result['response'][:100]}...")
    print(f"Confidence: {text_result['confidence']}")
    print(f"Success: {text_result['success']}")

    # Test image query
    image_query = "What does this brain MRI show? Is there a tumor?"
    image_result = process_medical_query(image_query, mock_image_data)
    print(f"\nImage Query: {image_query}")
    print(f"Response: {image_result['response'][:100]}...")
    print(f"Confidence: {image_result['confidence']}")
    print(f"Success: {image_result['success']}")

    print("\nMedical agent system tests completed.")