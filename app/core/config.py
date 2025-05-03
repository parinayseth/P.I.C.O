import os
from dotenv import load_dotenv
import logging
from pydantic_settings import BaseSettings




ENV = os.getenv('ENV', 'local')

if ENV == 'local':
    load_dotenv("local.env")
    
logging.info(f"Starting application in {ENV} environment")
    
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BUCKET_NAME=os.getenv("BUCKET_NAME")
BLOB_PATH=os.getenv("BLOB_PATH")
LLM_PROJECT_ID=os.getenv("LLM_PROJECT_ID")

LLM_REGION=os.getenv("LLM_REGION")
LLM_MODEL= "claude-3-5-sonnet@20240620"

class LogConfig(BaseSettings):
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")
    
    @property
    def log_level_enum(self):
        """Convert string log level to logging module constant"""
        return getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)

class MongoDBConfig(BaseSettings):
    MONGODB_URI: str = os.getenv("MONGODB_CLIENT_STRING", "mongodb://localhost:27017")
    # PATIENT_MONGODB_COLLECTION: str = os.getenv("PATIENT_MONGODB_COLLECTION", "patient_data")
    # DOCTOR_MONGODB_COLLECTION: str = os.getenv("DOCTOR_MONGODB_COLLECTION", "doctor_data")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "your_default_db")
    MONGODB_MAX_POOL_SIZE: int = int(os.getenv("MONGODB_MAX_POOL_SIZE", "10"))
    MONGODB_MIN_POOL_SIZE: int = int(os.getenv("MONGODB_MIN_POOL_SIZE", "1"))

class Settings(BaseSettings):
    APP_NAME: str = "PICO App"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    log: LogConfig = LogConfig()
    mongodb: MongoDBConfig = MongoDBConfig()

settings = Settings()

