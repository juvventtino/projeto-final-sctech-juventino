"""
 Ponto de entrada da API RESTful VisionGuard AI (FastAPI + YOLO11).

Execução local:
    uvicorn main:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.detector import (
    VISIONGUARD_CLASSES,
    InvalidImageError,
    detector_service,
)
from app.schemas import (
    ClassesResponse,
    DetectionResponse,
    HealthResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = PROJECT_ROOT / "models" / "benchmark_results.json"

app = FastAPI(
    title="VisionGuard AI — Object Detection API",
    description=(
        "Microserviço backend RESTful para detecção de objetos em imagens utilizando "
        "a arquitetura **YOLO11 (Ultralytics)** sobre o dataset multiclasse **VisionGuard-12** "
        "(12 classes de mobilidade urbana e objetos cotidianos)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Sistema"])
def root() -> dict[str, Any]:
    """Retorna informações gerais do microserviço e atalhos para a documentação."""
    return {
        "service": "VisionGuard AI — Object Detection API",
        "version": "1.0.0",
        "model": "YOLO11n",
        "dataset": "VisionGuard-12 (12 classes)",
        "endpoints": {
            "detect": "POST /detect",
            "classes": "GET /classes",
            "benchmark": "GET /benchmark",
            "health": "GET /health",
            "swagger_docs": "GET /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health_check() -> HealthResponse:
    """Verifica a disponibilidade da API e do modelo YOLO."""
    return HealthResponse(
        status="healthy",
        service="VisionGuard AI",
        model="YOLO11n",
        num_dataset_classes=len(VISIONGUARD_CLASSES),
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Dataset & Modelo"])
def list_classes() -> ClassesResponse:
    """Lista as 12 classes de objetos mapeadas no dataset VisionGuard-12."""
    return ClassesResponse(
        dataset="VisionGuard-12 (Urban & Daily Objects Subset - COCO128)",
        total_classes=len(VISIONGUARD_CLASSES),
        classes=VISIONGUARD_CLASSES,
    )


@app.get("/benchmark", tags=["Dataset & Modelo"])
def get_benchmark() -> dict[str, Any]:
    """Retorna os resultados do fine-tuning e o benchmark entre YOLO11n e YOLOv8n."""
    if not BENCHMARK_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Arquivo de resultados de benchmark não encontrado.",
        )
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


@app.post(
    "/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inferência YOLO"],
    summary="Detecta objetos em uma imagem enviada via upload (UploadFile)",
)
async def detect_objects(
    file: UploadFile = File(
        ...,
        description="Arquivo binário de imagem (JPEG, PNG, WEBP, BMP) para inferência.",
    ),
    confidence_threshold: float = Query(
        default=0.25,
        ge=0.05,
        le=0.95,
        description="Limiar mínimo de confiança (0.05 a 0.95) para incluir detecções.",
    ),
    only_project_classes: bool = Query(
        default=False,
        description="Se True, restringe o retorno estritamente às 12 classes do VisionGuard-12.",
    ),
) -> DetectionResponse:
    """
    Recebe um arquivo de imagem via `multipart/form-data` (`UploadFile`), executa a
    inferência com o modelo **YOLO11n** e retorna um JSON estruturado contendo a lista
    de objetos detectados (`class_name`, `confidence` e `box` com `x_min`, `y_min`, `x_max`, `y_max`).
    """
    image_bytes = await file.read()

    try:
        detections = detector_service.detect(
            image_bytes=image_bytes,
            confidence_threshold=confidence_threshold,
            only_project_classes=only_project_classes,
        )
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return DetectionResponse(
        status="success",
        detections=detections,
    )
