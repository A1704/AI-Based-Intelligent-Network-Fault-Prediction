"""
Main Application Entrypoint for NetFault-AI Framework.
Mounts FastAPI routes, static UI assets, and launches the web dashboard.
"""

import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from src.api.routes import router as api_router

app = FastAPI(
    title="NetFault-AI",
    description="AI-Based Intelligent Network Equipment Fault Prediction Framework",
    version="1.0.0"
)

# Static files and HTML templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "src", "web", "static")
templates_dir = os.path.join(BASE_DIR, "src", "web", "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)

# Include API Router
app.include_router(api_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the interactive NOC dashboard UI."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    print("=" * 70)
    print("  NetFault-AI: Network Equipment Fault Prediction Framework")
    print("  Interactive Dashboard: http://localhost:8000")
    print("  API Documentation:     http://localhost:8000/docs")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
