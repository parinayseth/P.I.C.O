import asyncio
import datetime
import logging
import os
import time
from typing import List
from fastapi import FastAPI, APIRouter, HTTPException, Form, File, UploadFile, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse

from app.db.helpers import download_documents, extract_text_from_pdf
from app.services.ai_services.helpers import get_followup_questions_ai, get_patient_summary



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
async def get_summary(request: Request):
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
            ai_response, visit_id, status_code = await get_patient_summary(user_id, qna, pdf_text, department_selected)
            if status_code == 200:
                return JSONResponse(content={'success': True, 'analysis': ai_response, 'visit_id': visit_id }, status_code=status_code)
            else:
                return JSONResponse(content={'success': False, 'error': ai_response, 'visit_id': visit_id}, status_code=status_code)
        except Exception as e:
            return JSONResponse(content={'error': f'Error getting AI response: {e}'}, status_code=500)

    except Exception as e:
        return JSONResponse(content={'error': f'An unexpected error occurred in getting Summary: {e}'}, status_code=500)


# @app.route('/get_summary', methods=["POST"])
# def get_summary():
#     try:
#         request_data = json.loads(request.data)
#         print("Request received:", request_data)

#         user_id = request_data.get('user_id')
#         qna = request_data.get('qna')
#         doc_id = request_data.get('doc_id')
#         department_selected  = request_data.get('department')
#         if len(doc_id) > 1:
#             try:
#                 download_file_path, status_code = database.download_azure_blob(doc_id, user_id)
#                 if status_code != 200:
#                     return jsonify({'success': False, 'error': download_file_path}), status_code

#                 pdf_text = helper.extract_text_from_pdf(download_file_path)
#                 # pdf_summary = helper.summarize_pdf_text(pdf_text)
#             except Exception as e:
#                 return jsonify({'error': f'Error processing PDF: {e}'}), 500
#         else:
#             pdf_text = ""

#         try:
#             ai_response, visit_id, status_code = helper.get_ai_response(user_id, qna, pdf_text, department_selected)
#             if status_code == 200:
#                 return jsonify({'success': True, 'analysis': ai_response, 'visit_id': visit_id }), status_code
#             else:
#                 return jsonify({'success': False, 'error': ai_response, 'visit_id': visit_id}), status_code
#         except Exception as e:
#             return jsonify({'error': f'Error getting AI response: {e}'}), 500

#     except Exception as e:
#         return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

