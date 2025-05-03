import logging
import os
import app.core.config as config
from app.services.gcp_services.helpers import _upload_file_to_bucket, _upload_to_bucket
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