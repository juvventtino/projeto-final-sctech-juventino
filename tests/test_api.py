"""
Testes automatizados de integração para a API VisionGuard AI (FastAPI + YOLO11)
e validação das 12 classes do dataset nas imagens variadas de teste.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

EXPECTED_12_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "dog",
    "backpack",
    "umbrella",
    "cup",
    "chair",
}


def test_health_endpoint() -> None:
    """Verifica se o endpoint /health responde corretamente."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "YOLO11n"
    assert data["num_dataset_classes"] >= 10
    assert data["num_dataset_classes"] == 12


def test_classes_endpoint_has_at_least_10_classes() -> None:
    """Garante que o sistema possui no mínimo 10 classes distintas (12 configuradas)."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 12
    returned_names = {item["class_name"] for item in data["classes"]}
    assert returned_names == EXPECTED_12_CLASSES


def test_benchmark_endpoint() -> None:
    """Verifica se os resultados do fine-tuning e benchmark estão disponíveis."""
    response = client.get("/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert data["champion_model"] == "YOLO11n"
    assert len(data["benchmark_comparison"]) == 2


def test_detect_endpoint_across_all_varied_samples() -> None:
    """
    Envia todas as 6 imagens variadas da pasta samples/ para POST /detect,
    valida o contrato JSON exigido e comprova a detecção das 12 classes do dataset.
    """
    sample_files = sorted(SAMPLES_DIR.glob("*.jpg"))
    assert len(sample_files) == 6, "Devem existir 6 imagens de teste em samples/"

    detected_project_classes: set[str] = set()

    for img_path in sample_files:
        with img_path.open("rb") as f:
            response = client.post(
                "/detect",
                params={"only_project_classes": True},
                files={"file": (img_path.name, f, "image/jpeg")},
            )

        assert response.status_code == 200, f"Falha na imagem {img_path.name}"
        payload = response.json()

        # Valida formato JSON exigido nas instruções (Seção 5.1)
        assert payload["status"] == "success"
        assert isinstance(payload["detections"], list)
        assert len(payload["detections"]) > 0, f"Nenhuma detecção em {img_path.name}"

        for det in payload["detections"]:
            assert "class_name" in det
            assert "confidence" in det
            assert "box" in det
            assert isinstance(det["class_name"], str)
            assert 0.0 <= det["confidence"] <= 1.0

            box = det["box"]
            assert {"x_min", "y_min", "x_max", "y_max"} == set(box.keys())
            assert all(isinstance(v, int) for v in box.values())
            assert box["x_max"] >= box["x_min"]
            assert box["y_max"] >= box["y_min"]

            if det["class_name"] in EXPECTED_12_CLASSES:
                detected_project_classes.add(det["class_name"])

    # Comprova que todas as 12 classes do dataset foram reconhecidas com sucesso
    assert detected_project_classes == EXPECTED_12_CLASSES


def test_detect_rejects_invalid_file() -> None:
    """Verifica tratamento de erro (HTTP 400) ao enviar arquivo inválido."""
    response = client.post(
        "/detect",
        files={"file": ("invalid.txt", b"conteudo nao e uma imagem", "text/plain")},
    )
    assert response.status_code == 400
