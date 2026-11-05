"""
Schemas Pydantic para validação de entrada/saída e documentação OpenAPI (Swagger)
do microserviço VisionGuard AI.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Coordenadas em pixels da caixa delimitadora (Bounding Box) do objeto detectado."""

    x_min: int = Field(..., description="Coordenada X inicial (esquerda) em pixels", examples=[150])
    y_min: int = Field(..., description="Coordenada Y inicial (topo) em pixels", examples=[200])
    x_max: int = Field(..., description="Coordenada X final (direita) em pixels", examples=[250])
    y_max: int = Field(..., description="Coordenada Y final (base) em pixels", examples=[300])


class DetectionItem(BaseModel):
    """Representa um único objeto detectado pelo modelo YOLO."""

    class_name: str = Field(
        ...,
        description="Nome da classe detectada pelo modelo YOLO",
        examples=["person"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nível de confiança da predição (0.0 a 1.0)",
        examples=[0.92],
    )
    box: BoundingBox = Field(..., description="Coordenadas do bounding box do objeto")


class DetectionResponse(BaseModel):
    """Payload JSON padrão retornado pelo endpoint POST /detect."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "detections": [
                    {
                        "class_name": "person",
                        "confidence": 0.92,
                        "box": {"x_min": 150, "y_min": 200, "x_max": 250, "y_max": 300},
                    },
                    {
                        "class_name": "car",
                        "confidence": 0.85,
                        "box": {"x_min": 40, "y_min": 50, "x_max": 100, "y_max": 120},
                    },
                ],
            }
        }
    )

    status: str = Field(default="success", description="Status do processamento da requisição")
    detections: list[DetectionItem] = Field(
        default_factory=list,
        description="Lista de objetos detectados na imagem enviada",
    )


class ClassItem(BaseModel):
    """Metadados de uma classe monitorada pelo dataset VisionGuard-12."""

    id: int
    coco_id: int
    class_name: str
    label_pt: str


class ClassesResponse(BaseModel):
    """Resposta do endpoint GET /classes listando as 12 classes do projeto."""

    dataset: str
    total_classes: int
    classes: list[ClassItem]


class HealthResponse(BaseModel):
    """Resposta do endpoint GET /health."""

    status: str
    service: str
    model: str
    num_dataset_classes: int
