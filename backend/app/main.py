"""FastAPI entrypoint for MachineGuard AI."""

from contextlib import asynccontextmanager
import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Path, Query, Response, status
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
    RandomForestTreeResponse,
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

# Configurable CORS origins with sensible defaults for development and Vercel deployments
ALLOWED_ORIGINS_ENV = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
)
allowed_origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
# If ALLOWED_ORIGINS contains "*", allow all origins (useful in dev/preview environments)
allow_all = "*" in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app" if not allow_all else None,
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


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["Monitoring"])
@app.api_route("/api/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["Monitoring"])
async def health(response: Response):
    service = ModelService.get_instance()
    is_ready = service.pipeline is not None
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=is_ready,
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


@app.get("/api/model-info", tags=["Model Info"])
async def get_model_info():
    service = ModelService.get_instance()
    meta = service.metadata
    metrics = service.metrics
    selected = metrics.get("selected_model", {})
    test_metrics = selected.get("test_metrics_tuned", {})
    splits = meta.get("dataset_statistics", {}).get("splits", {})
    return {
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "algorithm": meta.get("algorithm"),
        "total_estimators": meta.get("total_estimators", 150),
        "selected_threshold": service.threshold,
        "default_threshold": meta.get("default_threshold", 0.5),
        "test_samples": splits.get("test_records", 1500),
        "test_metrics": test_metrics,
        "baseline_comparison": metrics.get("baseline_comparison", {}),
        "cross_validation": metrics.get("cross_validation", {}),
        "feature_importances": meta.get("feature_importances", []),
        "dataset_statistics": meta.get("dataset_statistics", {}),
        "engineered_feature_definitions": meta.get("engineered_feature_definitions", {}),
        "ablation_comparison": metrics.get("ablation_comparison", {}),
        "data_honesty_statement": meta.get(
            "data_honesty_statement",
            "Educational prototype trained on the AI4I 2020 synthetic predictive-maintenance benchmark. Results are not validated for deployment on real industrial machinery.",
        ),
    }


@app.get(
    "/api/model/tree/{tree_index}",
    response_model=RandomForestTreeResponse,
    tags=["Model Inspection"],
)
@app.get(
    "/model/tree/{tree_index}",
    response_model=RandomForestTreeResponse,
    tags=["Model Inspection"],
)
async def get_model_tree(
    tree_index: int = Path(..., ge=0, description="0-indexed tree estimator number (0 to 149)"),
    max_depth: Optional[int] = Query(None, ge=1, le=20, description="Optional maximum depth filter for visualization"),
):
    service = ModelService.get_instance()
    try:
        return service.get_tree_structure(tree_index=tree_index, max_depth=max_depth)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree extraction error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(item: MachineInput):
    service = ModelService.get_instance()
    try:
        return service.predict(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
@app.post("/api/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch: BatchMachineInput):
    service = ModelService.get_instance()
    try:
        return service.predict_batch(batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
