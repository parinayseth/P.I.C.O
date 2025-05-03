import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class MongoDBConnection:
    client: Optional[AsyncIOMotorClient] = None
    db_name: str = settings.mongodb.MONGODB_DB
    
    def __init__(self, uri: str = None):
        """Initialize MongoDB connection settings"""
        self.mongodb_uri = uri or settings.mongodb.MONGODB_URI
        self.max_pool_size = settings.mongodb.MONGODB_MAX_POOL_SIZE
        self.min_pool_size = settings.mongodb.MONGODB_MIN_POOL_SIZE
        
    async def connect(self):
        """Create database connection."""
        try:
            if self.client is None:
                logger.info(f"Connecting to MongoDB at {self.mongodb_uri} (DB: {self.db_name})")
                
                self.client = AsyncIOMotorClient(
                    self.mongodb_uri,
                    maxPoolSize=self.max_pool_size,
                    minPoolSize=self.min_pool_size,
                    serverSelectionTimeoutMS=5000  # 5 second timeout
                )
                
                # Force a connection to verify it works
                await self.client.admin.command('ismaster')
                
                logger.info("Successfully connected to MongoDB")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def close(self):
        """Close database connection."""
        if self.client:
            logger.info("Closing MongoDB connection")
            self.client.close()
            self.client = None
            logger.info("MongoDB connection closed")

    def get_db(self):
        """Get database instance."""
        if self.client is None:
            raise ConnectionError("MongoDB client not initialized. Call connect() first.")
        return self.client[self.db_name]

    def get_collection(self, collection_name: str):
        """Get a specific collection."""
        return self.get_db()[collection_name]

# Create a global instance of the database connection
mongodb = MongoDBConnection()