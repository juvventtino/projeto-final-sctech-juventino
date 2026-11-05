"""
Script de Aquisição, Higienização e Preparação do Dataset (VisionGuard-12).

Extrai e prepara um dataset multiclasse no formato YOLO contendo 12 classes
distintas de objetos voltados para monitoramento urbano e ambientes cotidianos,
a partir do benchmark COCO128.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "coco128"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "visionguard12"
SAMPLES_DIR = PROJECT_ROOT / "samples"

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

# Mapeamento: ID original no COCO -> (Novo ID 0..11, Nome da Classe, Descrição PT-BR)
TARGET_CLASSES: dict[int, tuple[int, str, str]] = {
    0: (0, "person", "Pessoa / Pedestre"),
    1: (1, "bicycle", "Bicicleta"),
    2: (2, "car", "Carro"),
    3: (3, "motorcycle", "Motocicleta"),
    5: (4, "bus", "Ônibus"),
    7: (5, "truck", "Caminhão"),
    9: (6, "traffic light", "Semáforo"),
    16: (7, "dog", "Cachorro"),
    24: (8, "backpack", "Mochila"),
    25: (9, "umbrella", "Guarda-chuva"),
    41: (10, "cup", "Copo / Xícara"),
    56: (11, "chair", "Cadeira"),
}


def download_coco128() -> None:
    """Baixa e extrai o dataset COCO128 caso ainda não esteja presente localmente."""
    if (RAW_DIR / "images" / "train2017").exists():
        print(f"[INFO] Dataset base já encontrado em {RAW_DIR}")
        return

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR.parent / "coco128.zip"
    print(f"[INFO] Baixando COCO128 de {COCO128_URL}...")
    urllib.request.urlretrieve(COCO128_URL, zip_path)

    print("[INFO] Extraindo arquivo ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR.parent)
    zip_path.unlink(missing_ok=True)
    print("[INFO] Download e extração concluídos com sucesso.")


def prepare_visionguard12() -> dict:
    """Filtra, remapeia os rótulos para 12 classes (0..11) e organiza splits train/val."""
    download_coco128()

    src_images_dir = RAW_DIR / "images" / "train2017"
    src_labels_dir = RAW_DIR / "labels" / "train2017"

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    valid_records: list[tuple[Path, list[str], set[int]]] = []
    class_instance_counts: Counter[str] = Counter()
    class_image_counts: Counter[str] = Counter()

    for img_path in sorted(src_images_dir.glob("*.jpg")):
        label_path = src_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        remapped_lines: list[str] = []
        present_classes: set[int] = set()

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            orig_cls = int(parts[0])
            if orig_cls in TARGET_CLASSES:
                new_cls_id, cls_name, _ = TARGET_CLASSES[orig_cls]
                remapped_lines.append(f"{new_cls_id} {' '.join(parts[1:])}")
                present_classes.add(new_cls_id)
                class_instance_counts[cls_name] += 1

        if remapped_lines:
            for c_id in present_classes:
                cls_name = next(
                    name for _, (nid, name, _) in TARGET_CLASSES.items() if nid == c_id
                )
                class_image_counts[cls_name] += 1
            valid_records.append((img_path, remapped_lines, present_classes))

    # Copia todas as imagens filtradas para train e seleciona um subconjunto representativo
    # para validação cobrindo todas as 12 classes
    val_indices: set[int] = set()
    covered_classes: set[int] = set()

    for idx, (_, _, present_cls) in enumerate(valid_records):
        if len( present_cls - covered_classes ) > 0:
            val_indices.add(idx)
            covered_classes.update(present_cls)
        elif idx % 4 == 0:
            val_indices.add(idx)

    for idx, (img_path, remapped_lines, _) in enumerate(valid_records):
        # Treino recebe todas as amostras para maximizar aprendizado nas 12 classes
        dst_train_img = OUTPUT_DIR / "images" / "train" / img_path.name
        dst_train_lbl = OUTPUT_DIR / "labels" / "train" / f"{img_path.stem}.txt"
        shutil.copy2(img_path, dst_train_img)
        dst_train_lbl.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")

        if idx in val_indices:
            dst_val_img = OUTPUT_DIR / "images" / "val" / img_path.name
            dst_val_lbl = OUTPUT_DIR / "labels" / "val" / f"{img_path.stem}.txt"
            shutil.copy2(img_path, dst_val_img)
            dst_val_lbl.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")

    # Seleciona as 6 imagens com maior diversidade de classes para a pasta samples/
    ranked_samples = sorted(
        valid_records,
        key=lambda item: (len(item[2]), len(item[1])),
        reverse=True,
    )[:6]

    sample_files: list[dict] = []
    for i, (img_path, remapped_lines, present_cls) in enumerate(ranked_samples, start=1):
        cls_names = sorted(
            next(name for _, (nid, name, _) in TARGET_CLASSES.items() if nid == c_id)
            for c_id in present_cls
        )
        sample_name = f"sample_{i:02d}_{img_path.name}"
        dst_sample = SAMPLES_DIR / sample_name
        shutil.copy2(img_path, dst_sample)
        sample_files.append(
            {
                "file": f"samples/{sample_name}",
                "classes_present": cls_names,
                "num_annotations": len(remapped_lines),
            }
        )

    names_map = {
        new_id: cls_name
        for _, (new_id, cls_name, _) in sorted(
            TARGET_CLASSES.items(), key=lambda x: x[1][0]
        )
    }

    data_yaml = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names_map),
        "names": names_map,
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary = {
        "dataset_name": "VisionGuard-12 (Urban & Daily Objects Subset)",
        "source": "COCO128 (Ultralytics / MS-COCO)",
        "num_classes": len(names_map),
        "total_filtered_images": len(valid_records),
        "train_images": len(valid_records),
        "val_images": len(val_indices),
        "total_annotations": int(sum(class_instance_counts.values())),
        "classes": [
            {
                "id": new_id,
                "coco_id": coco_id,
                "name": cls_name,
                "description_pt": desc_pt,
                "instances": class_instance_counts[cls_name],
                "images_count": class_image_counts[cls_name],
            }
            for coco_id, (new_id, cls_name, desc_pt) in sorted(
                TARGET_CLASSES.items(), key=lambda x: x[1][0]
            )
        ],
        "samples": sample_files,
    }

    summary_path = PROJECT_ROOT / "datasets" / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[OK] Dataset preparado em {OUTPUT_DIR} com {summary['num_classes']} classes, "
        f"{summary['total_filtered_images']} imagens e {summary['total_annotations']} anotações."
    )
    return summary


if __name__ == "__main__":
    prepare_visionguard12()
