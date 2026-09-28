"""FastAPI entrypoint for MachineGuard AI."""

from contextlib import asynccontextmanager
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.model_service import ModelService
from app.schemas import (
    BatchMachineInput,
    BatchPredictionResponse,
    HealthResponse,
    MachineInput,
    ModelMetadataResponse,
    ModelMetricsResponse,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and pre-load model pipeline & metadata in memory at startup
    ModelService.get_instance()
    yield


app = FastAPI(
    title="MachineGuard AI API",
    description=(
        "AI-Based Predictive Maintenance of Industrial Machines Using Machine Learning. "
        "Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and development environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    service = ModelService.get_instance()
    return {
        "project": "MachineGuard AI",
        "description": "AI-Based Predictive Maintenance of Industrial Machines Using Machine Learning",
        "version": service.metadata.get("model_version", "1.0.0"),
        "status": "online",
        "data_honesty_statement": service.metadata.get(
            "data_honesty_statement",
            "Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark.",
        ),
        "docs_url": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    service = ModelService.get_instance()
    return HealthResponse(
        status="healthy",
        model_loaded=service.pipeline is not None,
        model_version=service.metadata.get("model_version", "1.0.0"),
        selected_threshold=service.threshold,
        data_honesty=service.metadata.get("data_honesty_statement", ""),
    )


@app.get("/metadata", response_model=ModelMetadataResponse, tags=["Model Info"])
async def get_metadata():
    service = ModelService.get_instance()
    return ModelMetadataResponse(metadata=service.metadata)


@app.get("/metrics", response_model=ModelMetricsResponse, tags=["Model Info"])
async def get_metrics():
    service = ModelService.get_instance()
    return ModelMetricsResponse(metrics=service.metrics)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(item: MachineInput):
    service = ModelService.get_instance()
    try:
        return service.predict(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch: BatchMachineInput):
    service = ModelService.get_instance()
    try:
        return service.predict_batch(batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
