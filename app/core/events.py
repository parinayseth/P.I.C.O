from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import (general_routes, ai_routes, db_routes, health_check)
import subprocess
import app.core.config as config
from app.services.gcp_services.helpers import download_embeddings_data
from app.db.session import mongodb

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("Starting application")
    # yield
    command = ["gcloud", "auth", "activate-service-account", f"--key-file={config.GOOGLE_APPLICATION_CREDENTIALS}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logger.error(f"Google Cloud Authentication Error: {stderr.decode()}")
        raise RuntimeError("Failed to authenticate with Google Cloud")
    else:
        logger.info("Successfully authenticated with Google Cloud")

    logger.info("Downloading Embeddings from GCP")
    download_embeddings_data()
    
    
    await mongodb.connect()

    yield
    
    
    # logging.info("Application is shutting down")