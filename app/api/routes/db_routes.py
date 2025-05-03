import asyncio
import datetime
import logging
import os
import time
from typing import List
import uuid
from fastapi import Depends, FastAPI, APIRouter, HTTPException, Form, File, UploadFile, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from app.db import helpers
from app.db.utils import get_collection
from app.core.config import settings

###### LOGGING SETUP ######
logger = logging.getLogger(__name__)
####### END OF LOGGING SETUP ######



####### API ROUTER SETUP #######
router = APIRouter(prefix="/db")


@router.get("/route_check", response_model=dict)
async def route_check():
    """
    Route check endpoint to verify if the API is running.
    """
    return JSONResponse(
        content={"status": "Success DB Route is running", "timestamp": datetime.datetime.utcnow().isoformat()},
        status_code=200
    )



@router.post("/upload_docs", response_model=dict)
async def store_docs(
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
    db_collection: AsyncIOMotorCollection = Depends(get_collection("patient_data")),

):
    """
    Endpoint to upload documents.
    """
    try:
        if not files:
            return JSONResponse(content={'error': 'No files provided'}, status_code=400)
        
        if not user_id:
            return JSONResponse(content={'error': 'User ID is required'}, status_code=400)

        for file in files:
            filename = f"{user_id}_{str(uuid.uuid4())}_{file.filename}"
            try:
                with open(filename, "wb") as f:
                    f.write(await file.read())
            except Exception as e:
                return JSONResponse(content={'error': f'Error saving file: {e}'}, status_code=500)

        # Simulate database upload
        response, status_code = await helpers.upload_documents(filename, user_id, db_collection)

        if status_code == 200:
            os.remove(filename)
            return JSONResponse(content={'success': True, 'path_stored': response}, status_code=status_code)
        else:
            return JSONResponse(content={'success': False, 'error': response}, status_code=status_code)

    except Exception as e:
        return JSONResponse(content={'error': f'An unexpected error occurred: {e}'}, status_code=500)
    
