from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import (general_routes, ai_routes, db_routes, health_check)
from app.core.events import lifespan
from app.core.logger import setup_logging
from app.core.config import settings
import logging
import uuid
import time
from fastapi.responses import JSONResponse

# Setup logging at application startup - use the log_config instance, not the class
setup_logging(log_level=settings.log.log_level_enum, log_file=settings.log.LOG_FILE)
logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware for logging
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request_id to context
    ctx = {"request_id": request_id}
    logger.info(f"Request started: {request.method} {request.url.path}", extra=ctx)
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Duration: {process_time:.3f}s",
            extra=ctx
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(
            f"Request failed: {request.method} {request.url.path} "
            f"- Duration: {process_time:.3f}s - Error: {str(e)}",
            extra=ctx
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id}
        )

app.include_router(health_check.router, prefix=settings.API_PREFIX)
app.include_router(general_routes.router, prefix=settings.API_PREFIX)
app.include_router(ai_routes.router, prefix=settings.API_PREFIX)
app.include_router(db_routes.router, prefix=settings.API_PREFIX)

# Log application startup
logger.info(f"Application '{settings.APP_NAME}' initialized")
