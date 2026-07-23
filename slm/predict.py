"""
Inferência do SLM (YOLOv8 fine-tuned) — Camada 1.

Carrega os pesos treinados no Kaggle (slm/weights/stride_yolov8s/best.pt) e
roda detecção de componentes de arquitetura em uma imagem de diagrama.

A saída de detect_components() é o formato de entrada compartilhado por:
    - stride.analyzer.analyze()               (heurística + contramedidas)
    - validation.claude_validator.validate()   (Camada 2)
    - validation.openai_validator.validate()   (Camada 2)
"""

from pathlib import Path
from typing import Optional

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "stride_yolov8s" / "best.pt"

# Cache simples em memória — evita recarregar os pesos (~22MB + inicialização
# do backbone) a cada requisição da API.
_model_cache: dict = {}


def _load_model(weights_path: Optional[str] = None):
    from ultralytics import YOLO  # import tardio: só carrega a dependência pesada quando necessário

    path = str(weights_path or DEFAULT_WEIGHTS)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Pesos do YOLOv8 não encontrados em '{path}'. "
            "Copie o best.pt gerado no treino do Kaggle para slm/weights/stride_yolov8s/."
        )
    if path not in _model_cache:
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


def detect_components(
    image_path: str,
    weights_path: Optional[str] = None,
    conf: float = 0.25,
) -> list[dict]:
    """
    Roda o YOLOv8 fine-tuned na imagem e devolve os componentes detectados.

    Args:
        image_path:    caminho da imagem do diagrama a analisar.
        weights_path:  caminho customizado para o best.pt
                       (default: slm/weights/stride_yolov8s/best.pt).
        conf:          confiança mínima (0-1) para manter uma detecção.

    Returns:
        Lista de dicts, um por componente detectado:
        {
            "component_id": str,   # único dentro da imagem, ex. "data_database_1"
            "node_name": str,      # igual a component_id (compatibilidade com stride/analyzer.py)
            "class_name": str,     # uma das 32 classes do data.yaml
            "label": str,          # versão legível, ex. "Data Database"
            "bbox": [x1, y1, x2, y2],  # coordenadas em pixels na imagem original
            "confidence": float,
        }
    """
    model = _load_model(weights_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)

    components = []
    counts_by_class: dict = {}

    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

            counts_by_class[class_name] = counts_by_class.get(class_name, 0) + 1
            component_id = f"{class_name}_{counts_by_class[class_name]}"

            components.append({
                "component_id": component_id,
                "node_name": component_id,
                "class_name": class_name,
                "label": class_name.replace("_", " ").title(),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(confidence, 4),
            })

    return components
