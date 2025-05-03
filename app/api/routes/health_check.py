from fastapi import APIRouter, Depends
from app.db.session import mongodb
import logging
from typing import Dict, Any
from pymongo.errors import ConnectionFailure
import time

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"], prefix="/health_check")

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify API and database status
    """
    result = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "api": {"status": "up"},
            "database": {"status": "unknown"}
        }
    }
    
    # Check database connection
    try:
        # Execute a simple command to check the database connection
        client = mongodb.client
        if client:
            await client.admin.command("ping")
            result["components"]["database"]["status"] = "up"
            logger.debug("Database health check: Connection successful")
        else:
            result["components"]["database"]["status"] = "down"
            result["status"] = "degraded"
            logger.warning("Database health check: No connection")
    except ConnectionFailure:
        result["components"]["database"]["status"] = "down"
        result["status"] = "degraded"
        logger.error("Database health check: Connection failed")
    except Exception as e:
        result["components"]["database"]["status"] = "error"
        result["status"] = "degraded"
        result["components"]["database"]["error"] = str(e)
        logger.exception(f"Database health check: Unexpected error: {str(e)}")
    
    return result