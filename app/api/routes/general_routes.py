import asyncio
import datetime
import logging
import os
import time
from typing import List
from fastapi import FastAPI, APIRouter, HTTPException, Form, File, UploadFile, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse



###### LOGGING SETUP ######
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
####### END OF LOGGING SETUP ######



####### API ROUTER SETUP #######
router = APIRouter(prefix="/general")
    
    
@router.get("/route_check", response_model=dict)
async def route_check():
    """
    Route check endpoint to verify if the API is running.
    """
    return JSONResponse(
        content={"status": "Success General Route is running", "timestamp": datetime.datetime.utcnow().isoformat()},
        status_code=200
    )




