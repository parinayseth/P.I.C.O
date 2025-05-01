from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import (general_routes, ai_routes, db_routes, health_check)
from app.core.events import lifespan


app = FastAPI(lifespan=lifespan)

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_check.router, prefix="/api")
app.include_router(general_routes.router, prefix="/api")
app.include_router(ai_routes.router, prefix="/api")
app.include_router(db_routes.router, prefix="/api")
