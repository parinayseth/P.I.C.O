import logging
import os
import uuid

from PyPDF2 import PdfReader
import app.core.config as config
from app.db.utils import get_collection
from app.services.gcp_services.helpers import _upload_file_to_bucket, _upload_to_bucket, download_from_bucket
from datetime import datetime, timedelta, timezone
logger = logging.getLogger(__name__)

def get_current_timestamp():
    return datetime.now(timezone.utc).isoformat()



async def upload_documents(file_path, user_id, db_collection):
    """
    Uploads documents to GCP Bucket and Update Path in MongoDB
    """
    try:
        blob_path = f"patients_data/{user_id}/{os.path.basename(file_path)}"
        _upload_file_to_bucket(config.BUCKET_NAME, blob_path, file_path)
        logger.info(f"Uploaded {file_path} to GCP bucket at {blob_path}")
        
        doc_update = {
            "path": blob_path,
            "timestamp": get_current_timestamp()
        }
        
        result = await db_collection.update_one(
            {"user_id": user_id},
            {"$push": {"docs_uploaded": doc_update}},
            upsert=True
        )
        
        if result.matched_count == 0:
            logger.info("MongoDB update result: Docs Updated")
        
        
        return blob_path, 200
    
    except Exception as e:
        return str(e), 500
    
async def download_documents(blod_path, patient_id):
    """
    Downloads documents from GCP Bucket to Local Folder
    """
    try:
        local_folder_path = os.path.join(os.getcwd(), "patients_data", patient_id)
        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path)
        
        download_path = os.path.join(local_folder_path, os.path.basename(blod_path))
        local_folder_path= download_from_bucket(config.BUCKET_NAME, blod_path, download_path)     

        logger.info(f"Downloaded {blod_path} to {local_folder_path}")
        return local_folder_path, 200
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return str(e), 500
    
async def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)

        num_pages = len(reader.pages)
        logger.info(f"Number of pages in PDF: {num_pages}")
        final_extracted = ""
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            text_on_page = f"Page {i + 1}:\n{text}\n"
            final_extracted += text_on_page


        return final_extracted, 200
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return "", 500
    
    
async def store_qna_response(user_id, qna, ai_response, validation_score, department_selected, db_collection):
    try:
        # Create the document to update
        appointment_data = {
            "qna": qna,
            "ai_response": ai_response,
            "validation_score": validation_score,
            "department_selected": department_selected
        }
        visit_id = user_id + str(uuid.uuid4())
        appointment = {
            "data": appointment_data,
            "timestamp": get_current_timestamp(),
            "visit_id": visit_id,
        }


        result = await db_collection.update_one(
            {"user_id": user_id},
            {"$push": {"appointments": appointment}},
            upsert=True
        )

        if result.matched_count == 0:
            logger.info("MongoDB update result: New document created")
        else:
            logger.info("MongoDB update result: Appointment data updated in existing document")

        return visit_id, 200
    except Exception as e:
        logger.error(f"Error updating MongoDB: {e}")
        return {"error": f"Error updating MongoDB: {e}"}, 500
    
    
# def store_qna_response(user_id, qna, ai_response, validation_score, department_selected):
# try:

#     # Create the document to update
#     appointment_data = {
#         "qna": qna,
#         "ai_response": ai_response,
#         "validation_score": validation_score,
#         "department_selected": department_selected
#     }
#     visit_id = user_id + str(uuid.uuid4())
#     appointment = {
#         "data": appointment_data,
#         "timestamp": helper.get_current_timestamp(),
#         "visit_id": visit_id,
#     }

#     result = patient_collection.update_one(
#         {"user_id": user_id},
#         {"$push": {"appointments": appointment}},
#         upsert=True
#     )

#     if result.matched_count == 0:
#         print(f"MongoDB update result: New document created")
#     else:
#         print(f"MongoDB update result: Appointment data updated in existing document")

#     return visit_id, 200
# except Exception as e:
#     print(f"Error updating MongoDB: {e}")
#     return {"error": f"Error updating MongoDB: {e}"}, 500