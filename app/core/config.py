import os
from dotenv import load_dotenv
import logging


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )



ENV = os.getenv('ENV', 'local')

if ENV == 'local':
    load_dotenv("local.env")
    
logging.info(f"Starting application in {ENV} environment")
    
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
