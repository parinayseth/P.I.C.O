import asyncio
import datetime
import logging
import os
import time
from typing import List
from fastapi import Depends, FastAPI, APIRouter, HTTPException, Form, File, UploadFile, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from app.db.helpers import download_documents, extract_text_from_pdf, get_doctor_details, get_visit_details
from app.db.utils import get_collection
from app.services.ai_services.helpers import doctor_mapping, get_followup_questions_ai, get_patient_summary, get_patient_summary_rag



###### LOGGING SETUP ######
logger = logging.getLogger(__name__)
####### END OF LOGGING SETUP ######



####### API ROUTER SETUP #######
router = APIRouter(prefix="/ai")


@router.get("/route_check", response_model=dict)
async def route_check():
    """
    Route check endpoint to verify if the API is running.
    """
    return JSONResponse(
        content={"status": "Success AI Route is running", "timestamp": datetime.datetime.utcnow().isoformat()},
        status_code=200
    )



@router.post("/get_followup_questions", response_model=dict)
async def get_followup_questions(request: Request):
    """
    Endpoint to get follow-up questions based on user input.
    """
    try:
        request_data = await request.json() 
        user_id = request_data.get('user_id')
        qna = request_data.get('qna')
        logger.info(f"Request received: {request_data}")

        try:
            ai_response, status_code = await get_followup_questions_ai(qna)
            if status_code == 200:
                return JSONResponse(content={'success': True, 'followup_questions': ai_response}, status_code=status_code)
            else:
                return JSONResponse(content={'success': False, 'error': ai_response}, status_code=status_code)
        except Exception as e:
            return JSONResponse(content={'error': f'Error getting Followup Questions: {e}'}, status_code=500)

    except Exception as e:
        return JSONResponse(content={'error': f'An unexpected error occurred in getting Followup Questions: {e}'}, status_code=500)



@router.post("/get_summary", response_model=dict)
async def get_summary(request: Request, 
                          db_collection: AsyncIOMotorCollection = Depends(get_collection("patient_data")),
):
    """
    Endpoint to get a summary based on user input and uploaded document.
    """
    try:
        request_data = await request.json()
        logger.info(f"Request received: {request_data}")

        user_id = request_data.get('user_id')
        qna = request_data.get('qna')
        doc_id = request_data.get('doc_id')
        department_selected  = request_data.get('department')
        
        if len(doc_id) > 1:
            try:
                download_file_path, status_code = await download_documents(doc_id, user_id)
                if status_code != 200:
                    return JSONResponse(content={'success': False, 'error': download_file_path}, status_code=status_code)

                pdf_text, text_status = extract_text_from_pdf(download_file_path)
            except Exception as e:
                return JSONResponse(content={'error': f'Error processing PDF: {e}'}, status_code=500)
        else:
            pdf_text = ""

        try:
            ai_response, visit_id, status_code = await get_patient_summary(user_id, qna, pdf_text, department_selected, db_collection)
            if status_code == 200:
                return JSONResponse(content={'success': True, 'analysis': ai_response, 'visit_id': visit_id }, status_code=status_code)
            else:
                return JSONResponse(content={'success': False, 'error': ai_response, 'visit_id': visit_id}, status_code=status_code)
        except Exception as e:
            return JSONResponse(content={'error': f'Error getting AI response: {e}'}, status_code=500)

    except Exception as e:
        return JSONResponse(content={'error': f'An unexpected error occurred in getting Summary: {e}'}, status_code=500)

@router.post("/rag_data", response_model=dict)
async def rag_data(request: Request, 
                          db_collection: AsyncIOMotorCollection = Depends(get_collection("patient_data")),
):
    """
    Endpoint to get a summary based on user input and uploaded document.
    """
    try:
        request_data = await request.json()
        logger.info(f"Request received: {request_data}")

        user_id = request_data.get('user_id')
        qna = request_data.get('qna')
        doc_id = request_data.get('doc_id')
        
        if len(doc_id) > 1:
            try:
                download_file_path, status_code = await download_documents(doc_id, user_id)
                if status_code != 200:
                    return JSONResponse(content={'success': False, 'error': download_file_path}, status_code=status_code)

                pdf_text, text_status = extract_text_from_pdf(download_file_path)
            except Exception as e:
                return JSONResponse(content={'error': f'Error processing PDF: {e}'}, status_code=500)
        else:
            pdf_text = ""

        try:
            ai_response, status_code = await get_patient_summary_rag(qna)
            if status_code == 200:
                return JSONResponse(content={'success': True, 'analysis': ai_response }, status_code=status_code)
            else:
                return JSONResponse(content={'success': False, 'error': ai_response}, status_code=status_code)
        except Exception as e:
            return JSONResponse(content={'error': f'Error getting AI response: {e}'}, status_code=500)

    except Exception as e:
        return JSONResponse(content={'error': f'An unexpected error occurred in getting Summary: {e}'}, status_code=500)


@router.post("/map_doctor", response_model=dict)
async def mapping_doctor(request: Request,
                        pateint_db_collection: AsyncIOMotorCollection = Depends(get_collection("patient_data")),
                        doctor_db_collection: AsyncIOMotorCollection = Depends(get_collection("doctor_data") ),
                        ):
    try:
        request_data = await request.json()
        logger.info(f"Request received: {request_data}")
        
        visit_id = request_data.get('visit_id')
        visit_details, status_code = await get_visit_details(visit_id, pateint_db_collection)
        
        if status_code != 200:
            return JSONResponse(content={'success': False, 'error': visit_details}, status_code=status_code)
        
        department = visit_details.get('data', {}).get('department_selected')
        logger.info(f"Department: {department}")
        
        patient_age = visit_details.get('data', {}).get('qna', {}).get('What is your age and Sex?')
        logger.info(f"Patient Age: {patient_age}")
        
        about_patient = visit_details.get('data', {}).get('qna', {})
        logger.info(f"About Patient: {about_patient}")
        
        doctors_list, status_code = await get_doctor_details(department,doctor_db_collection)
        logger.info(f"Doctors List: {doctors_list}")
        
        get_doctor = await doctor_mapping(dept=department, AGE=patient_age, about_patient= about_patient, available_doctors=doctors_list)
        
        logger.info(f"Doctor Mapping: {get_doctor}")
        
        if "Doctor's User ID" in get_doctor:
            doctor_user_id = get_doctor["Doctor's User ID"]
            
            # Find the doctor details by user_id
            mapped_doctor = None
            for doctor in doctors_list:
                if doctor['user_id'] == doctor_user_id:
                    mapped_doctor = doctor
                    break
            
            if mapped_doctor:
                response = {
                    'success': True,
                    'patient_department': department,
                    'doctor_mapped': {
                        'Doctor_user_id': doctor_user_id,
                        'Doctor_name': f"{mapped_doctor['first_name']} {mapped_doctor['last_name']}",
                        'Doctor_email': mapped_doctor['email'],
                        'Day of Appointment': get_doctor.get('Next Workday', 'Not specified'),
                        'Appointment Duration': get_doctor.get('OPD Timing', 'Not specified')
                    }
                }
                return JSONResponse(content=response, status_code=status_code)
            else:
                return JSONResponse(content={'success': False, 'error': f"Doctor with user_id {doctor_user_id} not found"}, status_code=404)
        else:
            return JSONResponse(content={'success': False, 'error': "Doctor's User ID not found in mapping result"}, status_code=400)
    except Exception as e:
        logger.error(f"Error in mapping doctor: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
        