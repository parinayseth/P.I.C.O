from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import (general_routes, ai_routes, db_routes, health_check)
import subprocess
import logging
import os
import app.core.config as config



@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logging.info("Starting application")
    # command = ["gcloud", "auth", "activate-service-account", f"--key-file={config.GOOGLE_APPLICATION_CREDENTIALS}"]
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = process.communicate()
    
    # if process.returncode != 0:
    #     logging.error(f"Google Cloud Authentication Error: {stderr.decode()}")
    #     raise RuntimeError("Failed to authenticate with Google Cloud")
    # else:
    #     logging.info("Successfully authenticated with Google Cloud")

    # yield
    
    
    # logging.info("Application is shutting down")