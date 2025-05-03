from contextlib import asynccontextmanager
import logging
import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.api.routes import (general_routes, ai_routes, db_routes, health_check)
import subprocess
import app.core.config as config
from app.services.ai_services import resources
from app.services.gcp_services.helpers import download_embeddings_data
from app.db.session import mongodb

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")
    
    # Google Cloud authentication
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
    
    # Load resources once during startup
    try:
        logger.info("Loading FAISS index, embedding model, and dataset...")
        faiss_output = "embeddings/KnowledgeBase.faiss"
        csv_output = "embeddings/MedMCQA_Combined_DF.csv"
        
        # Load into the imported resources module
        resources.faiss_index = faiss.read_index(faiss_output)
        resources.embedding_model = SentenceTransformer("hkunlp/instructor-xl")
        resources.med_mcqa_df = pd.read_csv(csv_output)
        
        logger.info("Successfully loaded all resources")
    except Exception as e:
        logger.error(f"Error loading resources: {e}")
        raise
    
    await mongodb.connect()
    
    yield
    
    # Cleanup code if needed
    logger.info("Shutting down and cleaning up resources")
    
    # logging.info("Application is shutting down")