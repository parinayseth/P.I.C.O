import logging
import os
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud.exceptions import NotFound
import app.core.config as config
logger = logging.getLogger(__name__)


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/parineyseth/Documents/personal-projects/pico/P.I.C.O/crendentials.json"

def _upload_to_bucket(bucket_name, blob_path, local_folder_path):
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    for root, dirs, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            blob_name = os.path.relpath(local_file_path, local_folder_path)
            blob_name = os.path.join(blob_path, blob_name).replace("\\", "/")
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
           
def _upload_file_to_bucket(bucket_name, blob_path, local_file_path):
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file_path)
     
            
def download_from_bucket(bucket_name, blob_path, local_folder_path):
    """
    Downloads files from a Google Cloud Storage bucket to a local folder.
    """
    bucket = storage.Client().bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=blob_path)
    for blob in blobs:
        if blob.name.endswith("/"):  
            continue
        relative_path = blob.name[len(blob_path) :].lstrip("/")
        local_file_path = os.path.join(local_folder_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
    return local_folder_path



def download_embeddings_data():
    """
    Downloads embeddings data from Google Cloud Storage to a local folder.
    """
    bucket_name = config.BUCKET_NAME
    blob_path = config.BLOB_PATH
    local_folder_path = os.path.join(os.getcwd(), "embeddings")
    
    try:
        if os.path.exists(local_folder_path):
            logger.info("Embeddings data already exists")
            return
        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path)
            
        download_from_bucket(bucket_name, blob_path, local_folder_path)
        logger.info(f"Downloaded embeddings data to {local_folder_path}")
    except NotFound:
        logger.error(f"Bucket {bucket_name} or blob {blob_path} not found.")