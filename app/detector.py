"""
Módulo de Inferência de Visão Computacional com YOLO11 (VisionGuard AI).

Encapsula o carregamento do modelo campeão (YOLO11n) e a lógica de detecção
de objetos multiclasse, convertendo saídas matriciais em estruturas JSON limpas.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_config")

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
FINETUNED_WEIGHTS_PATH = PROJECT_ROOT / "models" / "visionguard12_yolo11n_finetuned.pt"

# As 12 classes oficiais do dataset VisionGuard-12
VISIONGUARD_CLASSES: list[dict[str, Any]] = [
    {"id": 0, "coco_id": 0, "class_name": "person", "label_pt": "Pessoa / Pedestre"},
    {"id": 1, "coco_id": 1, "class_name": "bicycle", "label_pt": "Bicicleta"},
    {"id": 2, "coco_id": 2, "class_name": "car", "label_pt": "Carro"},
    {"id": 3, "coco_id": 3, "class_name": "motorcycle", "label_pt": "Motocicleta"},
    {"id": 4, "coco_id": 5, "class_name": "bus", "label_pt": "Ônibus"},
    {"id": 5, "coco_id": 7, "class_name": "truck", "label_pt": "Caminhão"},
    {"id": 6, "coco_id": 9, "class_name": "traffic light", "label_pt": "Semáforo"},
    {"id": 7, "coco_id": 16, "class_name": "dog", "label_pt": "Cachorro"},
    {"id": 8, "coco_id": 24, "class_name": "backpack", "label_pt": "Mochila"},
    {"id": 9, "coco_id": 25, "class_name": "umbrella", "label_pt": "Guarda-chuva"},
    {"id": 10, "coco_id": 41, "class_name": "cup", "label_pt": "Copo / Xícara"},
    {"id": 11, "coco_id": 56, "class_name": "chair", "label_pt": "Cadeira"},
]

COCO_ID_TO_CLASS_NAME: dict[int, str] = {
    item["coco_id"]: item["class_name"] for item in VISIONGUARD_CLASSES
}
TARGET_COCO_IDS: list[int] = list(COCO_ID_TO_CLASS_NAME.keys())


class InvalidImageError(ValueError):
    """Exceção lançada quando o arquivo enviado não é uma imagem válida."""


class ObjectDetector:
    """Serviço singleton para detecção de objetos utilizando arquitetura YOLO11."""

    def __init__(self, weights_path: Path | None = None) -> None:
        self.weights_path = weights_path or DEFAULT_WEIGHTS_PATH
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """Carrega o modelo YOLO sob demanda (lazy loading) e mantém em memória."""
        if self._model is None:
            self._model = YOLO(str(self.weights_path))
        return self._model

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.25,
        only_project_classes: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Processa os bytes de uma imagem e retorna a lista de detecções estruturadas.

        Args:
            image_bytes: Conteúdo binário da imagem enviada via upload.
            confidence_threshold: Limiar mínimo de confiança (padrão: 0.25).
            only_project_classes: Se True, restringe estritamente às 12 classes do dataset.

        Returns:
            Lista de dicionários contendo `class_name`, `confidence` e `box`.
        """
        if not image_bytes:
            raise InvalidImageError("O arquivo enviado está vazio.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(
                "Não foi possível decodificar o arquivo enviado como uma imagem válida."
            ) from exc

        classes_filter = TARGET_COCO_IDS if only_project_classes else None
        results = self.model.predict(
            source=image,
            conf=confidence_threshold,
            classes=classes_filter,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        if not results:
            return detections

        result = results[0]
        names_map = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = round(float(box.conf[0].item()), 2)
            x_min, y_min, x_max, y_max = [int(round(coord)) for coord in box.xyxy[0].tolist()]
            class_name = COCO_ID_TO_CLASS_NAME.get(cls_id, names_map.get(cls_id, str(cls_id)))

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    },
                }
            )

        return detections


detector_service = ObjectDetector()
