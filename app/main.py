from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.database import engine, Base
from app.api.routes import router
from app.core.config import settings

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Object Counter",
    description="Open-vocabulary object counting API",
    version="1.0.0",
)

# CORS - allow all for simple web usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve frontend
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


@app.on_event("startup")
async def startup():
    print("=" * 50)
    print("AI Object Counter is running!")
    print(f"Upload dir: {settings.UPLOAD_DIR}")
    print(f"Model: {settings.MODEL_NAME}")
    print("=" * 50)
