import datetime
import json
import logging
import pandas as pd

import faiss
import pytz
from sentence_transformers import SentenceTransformer
from app.db.helpers import store_qna_response
from app.services.ai_services.ai_agents import process_medical_consultation
from app.services.ai_services.gemini_services import gemini_call_flash_2
from prompts.system_prompts import followup_qna, get_validation_prompt, mapping_prompt, patient_info
from app.services.ai_services import resources


logger = logging.getLogger(__name__)

MAX_TRIES = 3
async def get_followup_questions_ai(qna):
    try:
        questioning_prompt = followup_qna()
        qna_string = ""
        for question, answer in qna.items():
            qna_string += f"'question': '{question}' \n 'Answer': '{answer}'\n"

        
        
        text_list, status = await gemini_call_flash_2(system_prompt=questioning_prompt, user_prompt=qna_string, user_feedback=None, model_name="gemini-2.0-flash-001")
        logger.info(f"Gemini response: {text_list}")
        
        formatted_list = await extract_response_from_delimeters(text_list, extraction_key="json")
        if formatted_list is None:
            logger.error("Failed to extract JSON response from Gemini API.")
            return "Error extracting JSON response", 500
        formatted_json = json.loads(formatted_list)
        
        logger.info(f"Formatted follow-up questions: {formatted_list}")
        return formatted_json, 200

    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        return "Error processing the request", 500
    
    
async def format_followup_qna_response(text_list):
    try:
        questions = []
        for text in text_list:
            if '$$$' in text:
                parts = text.split('$$$')
                if len(parts) > 1:
                    questions_part = parts[1].strip()
                    questions_list = questions_part.split('\n')
                    for question in questions_list:
                        question = question.strip()
                        if question:
                            questions.append(question)
        return questions
    except Exception as e:
        logger.error(f"An error occurred while extracting questions: {e}")
        return []
    
async def extract_response_from_delimeters(ai_response, extraction_key="response"):
        """
        Extract the response from the response string.
        Parameters: ai_response (str): The response string.
        Returns: response (str): The extracted response content.
        """
        json_code_start = f"```{extraction_key}"
        json_code_end = "```"
        start_index = ai_response.find(json_code_start) + len(json_code_start)
        end_index = ai_response.find(json_code_end, start_index)

        if start_index != -1 and end_index != -1:
            json_response = ai_response[start_index:end_index].strip()
            # print(f"Extracted JSON response: {json_response}")
            return json_response
        else:
            return None
        
        
async def get_patient_summary(user_id, qna, doc_summary, department_selected,db_collection):
    try:
        data = {"qna": qna}
        goal = await extract_and_format_qna(qna)
        logger.info(f"Goal: {goal}")
        # information = "Relevant documents from the Vector database including previous case studies and medical guidelines on hypertension and headaches"  # TODO: Adding Vector DB Function
        about = "\n".join([f"Question: {question}\nAnswer: {answer}\n" for question, answer in qna.items()])
        logger.info(f"About: {about}")
        information = await query_answer([goal])
        logger.info(f"Information: {information}")
        
        
        system_prompt = patient_info()
        user_prompt = f"\nGenerate a summary diagnosis based upon these {goal}, {about}, {doc_summary}, {information}"
        ai_response, status = await gemini_call_flash_2(system_prompt=system_prompt, user_prompt=user_prompt, user_feedback=None, model_name="gemini-2.0-flash-001")
        logger.info(f"AI Response: {ai_response}")

        
        validation_sytem_prompt = get_validation_prompt(ai_response, goal, about, doc_summary, information)
        validation_result, status = await gemini_call_flash_2(system_prompt=validation_sytem_prompt, user_prompt=ai_response, user_feedback=None, model_name="gemini-2.0-flash-001")
        logger.info(f"Validation Result: {validation_result}")
        
        confidence_score = await extract_confidence_score(validation_result)
        logger.info(f"Confidence Score: {confidence_score}")

        store_response, status_code = await store_qna_response(user_id, qna, ai_response, confidence_score, department_selected, db_collection)
        if status_code != 200:
            logger.error(f"Failed to store QnA response: {store_response}")
            return "Error storing QnA response", 500
        
        # ai_response = ""
        return ai_response, store_response, 200

    except Exception as e:
        logger.error(f"An error occurred while processing AI response: {e}")
        return "Error processing AI response", 500


async def run_consultation(patient_data, qna_responses,doc_summary, information):
    """
    Run a mock medical consultation with realistic test data.
    """
    print("Starting medical consultation...\n")

    # Process the consultation
    results = await process_medical_consultation(patient_data, qna_responses,doc_summary, information)

    # Print results
    logger.info(f"Consultation Results: {results}")
    # print(results["formatted"])
    # logger.info(f"Formatted Results: {results.get('formatted', 'No formatted results available')}")
    # Print additional details if specialist was needed
    if results["raw"]["specialist_needed"]:
        print(f"\nSpecialist Referral: {results['raw']['specialist_needed']}")

    formatted_result = results.get("formatted",None)
    return formatted_result

async def get_patient_summary_v2(user_id, qna, doc_summary, department_selected, db_collection):
    try:
        # data = {"qna": qna}
        goal = await extract_and_format_qna(qna)
        logger.info(f"Goal: {goal}")

        about = "\n".join([f"Question: {question}\nAnswer: {answer}\n" for question, answer in qna.items()])
        logger.info(f"About: {about}")
        information = await query_answer([goal])
        # logger.info(f"Information: {information}")
        

        patient_summary = await run_consultation(goal, about, doc_summary, information)

        updated_rag_information = await query_answer([patient_summary])
        if not updated_rag_information:
            updated_rag_information = information

        system_prompt = patient_info()
        user_prompt = f"\nGenerate a summary diagnosis based upon these Patient Details \n\n Patient's Agenda = {goal} \n\n About Patient \n {about} \n\n Reference Documents Provided by Patient = \n {doc_summary}, Past Case Studies = \n {information}" if not patient_summary else f"\nGenerate a summary diagnosis based these Patient Details \n\n Patient's Agenda = {goal} \n\n About Patient \n {patient_summary} \n\n Past Case Studies = \n {updated_rag_information}"
        
        # logger.info(f"User Prompt: {user_prompt}")
        
        
        ai_response, status = await gemini_call_flash_2(system_prompt=system_prompt, user_prompt=user_prompt,
                                                        user_feedback=None, model_name="gemini-2.0-flash-001")
        # logger.info(f"AI Response: {ai_response}")

        validation_sytem_prompt = get_validation_prompt(ai_response, goal, about, doc_summary, information)
        validation_result, status = await gemini_call_flash_2(system_prompt=validation_sytem_prompt,
                                                              user_prompt=ai_response, user_feedback=None,
                                                              model_name="gemini-2.0-flash-001")
        logger.info(f"Validation Result: {validation_result}")

        confidence_score = await extract_confidence_score(validation_result)
        logger.info(f"Confidence Score: {confidence_score}")

        store_response, status_code = await store_qna_response(user_id, qna, ai_response, confidence_score,
                                                               department_selected, db_collection)
        if status_code != 200:
            logger.error(f"Failed to store QnA response: {store_response}")
            return "Error storing QnA response", 500

        # ai_response = ""
        return ai_response, store_response, 200

    except Exception as e:
        logger.error(f"An error occurred while processing AI response: {e}")
        return "Error processing AI response", 500


async def extract_confidence_score(validation_result):
    try:
        start = validation_result.find("$$$")
        end = validation_result.find("$$$", start + 3)
        if start != -1 and end != -1:
            score = validation_result[start + 3:end]
            return score.strip()
        else:
            raise ValueError("Confidence score delimiters not found in the validation result.")
    except Exception as e:
        logger.error(f"An error occurred while extracting the confidence score: {e}")
        return "0"  # Return a default score or handle appropriately

async def extract_and_format_qna(json_data):
    try:
        keys_to_include = [
            "Purpose of Visit",
            "What is your age and Sex?",
            "What are your main symptoms and for how long you have been experiencing?",
            "Are you currently taking any medications? If so, what are they?",
            "Are there any hereditary conditions in your family?",
            "Do you have any known allergies?"
        ]
        complete_string = ""
        for key in keys_to_include:
            complete_string += f"{key} : {json_data[key]} "
        
        return complete_string

    except Exception as e:
        print(f"An error occurred while extracting and formatting Q&A: {e}")
        return [], ""
    
    
# faiss_output = "embeddings/KnowledgeBase.faiss"
# csv_output = "embeddings/MedMCQA_Combined_DF.csv"
# index_path = faiss_output
# index = faiss.read_index(index_path)
# Embeddingmodel = SentenceTransformer("hkunlp/instructor-xl")
# MedMCQA_Combined_DF = pd.read_csv(csv_output)
# index = faiss.read_index(index_path)

async def query_answer(query):
    try:
        # Use the globally loaded resources
        # logger.info(f"Querying with: {query}")
        # logger.info(f"Embedding model: {resources.embedding_model}")
        
        query_embeddings_without_reshape = resources.embedding_model.encode(query)
        # logger.info(f"Query embeddings without reshape: {query_embeddings_without_reshape}")
        
        query_embedding_base = query_embeddings_without_reshape[0]
        # logger.info(f"Query embedding base: {query_embedding_base}")
        
        query_embedding = query_embedding_base.reshape(1, -1)
        # logger.info(f"Query embedding reshaped: {query_embedding}")
        # query_embedding = embedding_model.encode([query])[0].reshape(1, -1)
        top_k = 3
        scores, index_vals = resources.faiss_index.search(query_embedding, top_k)
        
        extracted_rag_data = resources.med_mcqa_df['question_exp'].loc[list(index_vals[0])].to_list()

        # Join them into a single string with a header
        extracted_rag_str = f"Previous Case Studies:\n{' '.join(extracted_rag_data)}"
        
        logger.info(f"For query '{query}', \n retrieved {extracted_rag_str} documents \n\n")
        
        return extracted_rag_str
    except Exception as e:
        logger.error(f"An error occurred while querying the answer: {e}")
        return []
    
async def hyde_query_answer(query):
    """
    Implements the Hypothetical Document Embeddings (HyDE) RAG approach.
    This method:
    1. Generates a hypothetical document that would answer the query
    2. Uses that synthetic document for embedding and retrieval instead of the original query
    3. Returns the retrieved documents
    
    Args:
        query (list): List containing the user query
    
    Returns:
        list: Retrieved documents from the knowledge base
    """
    try:
        
        # Step 1: Generate a hypothetical document using Gemini
        system_prompt = """You are a medical professional tasked with creating a detailed response that would answer the following medical query. 
        Create a comprehensive and informative response that covers the key medical aspects of this query.
        Focus on providing factual information that would be useful for retrieval from a medical knowledge base.
        Keep your response focused and relevant to the query."""
        
        user_prompt = query[0] if isinstance(query, list) else query
        
        hypothetical_doc, status = await gemini_call_flash_2(
            system_prompt=system_prompt, 
            user_prompt=user_prompt, 
            user_feedback=None, 
            model_name="gemini-2.0-flash-001",
            temp = 0.7
            )
        
        logger.info(f"Generated hypothetical document: {hypothetical_doc}")
        
        if status != 200 or not hypothetical_doc:
            logger.warning("Failed to generate hypothetical document, falling back to original query")
            return await query_answer([query])
        
        # Step 2: Embed the hypothetical document instead of the original query
        hypothetical_embedding = resources.embedding_model.encode([hypothetical_doc])
        
        # Step 3: Retrieve relevant documents using the hypothetical embedding
        top_k = 5 
        scores, index_vals = resources.faiss_index.search(hypothetical_embedding, top_k)
        
        # Step 4: Filter results to return most relevant
        retrieved_docs = resources.med_mcqa_df['question_exp'].loc[list(index_vals[0])].to_list()
        
        logger.info(f"HyDE RAG retrieved {len(retrieved_docs)} documents")
        
        return retrieved_docs[:3]  # Return top 3 to maintain consistency with original query_answer
    
    except Exception as e:
        logger.error(f"An error occurred in hyde_query_answer: {e}")
        logger.info("Falling back to standard query_answer method")
        return await query_answer([query])

# Updated get_patient_summary function that uses hyde_query_answer
async def get_patient_summary_with_hyde(user_id, qna, doc_summary, department_selected, db_collection, use_hyde=True):
    try:
        data = {"qna": qna}
        goal = await extract_and_format_qna(qna)
        logger.info(f"Goal: {goal}")
        
        about = "\n".join([f"Question: {question}\nAnswer: {answer}\n" for question, answer in qna.items()])
        logger.info(f"About: {about}")
        
        # Use Hyde RAG or standard RAG based on the parameter
        if use_hyde:
            information = await hyde_query_answer([goal])
            logger.info(f"Hyde RAG Information: {information}")
        else:
            information = await query_answer([goal])
            logger.info(f"Standard RAG Information: {information}")
        
        system_prompt = patient_info()
        user_prompt = f"\nGenerate a summary diagnosis based upon these {goal}, {about}, {doc_summary}, {information}"
        ai_response, status = await gemini_call_flash_2(
            system_prompt=system_prompt, 
            user_prompt=user_prompt, 
            user_feedback=None, 
            model_name="gemini-2.0-flash-001"
        )
        logger.info(f"AI Response: {ai_response}")

        validation_system_prompt = get_validation_prompt(ai_response, goal, about, doc_summary, information)
        validation_result, status = await gemini_call_flash_2(
            system_prompt=validation_system_prompt, 
            user_prompt=ai_response, 
            user_feedback=None, 
            model_name="gemini-2.0-flash-001"
        )
        logger.info(f"Validation Result: {validation_result}")
        
        confidence_score = await extract_confidence_score(validation_result)
        logger.info(f"Confidence Score: {confidence_score}")

        store_response, status_code = await store_qna_response(
            user_id, 
            qna, 
            ai_response, 
            confidence_score, 
            department_selected, 
            db_collection,
            retrieval_method="hyde_rag" if use_hyde else "standard_rag"  # Store which method was used
        )
        
        if status_code != 200:
            logger.error(f"Failed to store QnA response: {store_response}")
            return "Error storing QnA response", 500
        
        return ai_response, store_response, 200

    except Exception as e:
        print(f"An error occurred while querying the answer: {e}")
        return []
    
    
async def doctor_mapping(dept, AGE, about_patient, available_doctors):
    """
    Implements the Doctor Mapping system.
    This method:
    1. Takes in patient details and available doctors
    2. Maps the most suitable doctor to the patient based on the given information
    3. Returns the recommended doctor's information in JSON format
    
    Args:
        dept (str): Department of the patient
        AGE (int): Age of the patient
        about_patient (str): Symptoms of the patient
        available_doctors (list): List of available doctors
    
    Returns:
        dict: Recommended doctor's information in JSON format
    """
    try:
        today = datetime.date.today()
        day_of_week = today.strftime("%A")
        utc_time = datetime.datetime.utcnow()
        india_time_zone = pytz.timezone('Asia/Kolkata')
        local_time = utc_time.astimezone(india_time_zone)
        currtime = local_time.strftime("%I:%M %p")
        docs = ""
        for doctor in available_doctors:
            docs += f"user_id: {doctor['user_id']}, Designation: {doctor['department']}, working_days: {doctor['working_days']} appointments_bookend: {doctor['appointments']}" + "\n"
            

        for doctor in available_doctors:
            docs += f"user_id: {doctor['user_id']}, Designation: {doctor['department']}, working_days: {doctor['working_days']} appointments_bookend: {doctor['appointments']}" + "\n"
            
        system_mapping_prompt = mapping_prompt(dept, docs, AGE, about_patient, currtime, day_of_week)
        
        combined = f"\nMap an OPD session for this {dept}, {AGE}, {about_patient}, \n{docs}"

        ai_response, status = await gemini_call_flash_2(system_prompt=system_mapping_prompt, user_prompt=combined, user_feedback=None, model_name="gemini-2.0-flash-001")
        
        extracted_response = await extract_response_from_delimeters(ai_response, extraction_key="json")
        if extracted_response is None:
            logger.error("Failed to extract JSON response from Gemini API.")
            return "Error extracting JSON response", 500
        
        extracted_json = json.loads(extracted_response)
        logger.info(f"Formatted doctor mapping response: {extracted_json}")
        return extracted_json, 200

    except Exception as e:
        logger.error(f"An error occurred while processing the doctor mapping: {e}")
        return {"error": "Error processing the doctor mapping"}, 500
