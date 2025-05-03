import logging
from fastapi import Depends, HTTPException, status
from app.db.session import mongodb, MongoDBConnection
from motor.motor_asyncio import AsyncIOMotorCollection
from typing import Callable, Dict, Type

logger = logging.getLogger(__name__)

class DatabaseNotConnectedError(Exception):
    """Raised when a database operation is attempted without a valid connection"""
    pass

def get_collection(collection_name: str) -> Callable:
    """
    Creates a dependency that provides access to a specific MongoDB collection
    
    Usage:
        @router.get("/users")
        async def get_users(users_collection: AsyncIOMotorCollection = Depends(get_collection("users"))):
            result = await users_collection.find().to_list(length=100)
            return result
    """
    async def _get_collection() -> AsyncIOMotorCollection:
        if mongodb.client is None:
            logger.error(f"Attempted to access collection '{collection_name}' before database connection")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection not available"
            )
        
        return mongodb.get_collection(collection_name)
    
    return _get_collection

# Dictionary of collection names to collection types if you want to use typed collections
collection_types: Dict[str, Type] = {
    "patient_data": dict,  # Replace with your actual types if needed
    "doctors_data": dict,
    # Add additional collection mappings as needed
}

def get_database():
    """
    Dependency that provides access to the MongoDB database object
    """
    if mongodb.client is None:
        logger.error("Attempted to access database before connection")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )
    
    return mongodb.get_db()