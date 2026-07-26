"""
Detecção de fluxos de dados (setas) — extensão da Camada 1.

Usa o modelo YOLO11-pose publicado pelo autor do dataset de componentes
(guillherms/vision-architecture-analyzer-yolo11-pose, Hugging Face), treinado
no dataset stride-architecture-flows-v1: cada seta do diagrama vira uma
detecção com 2 keypoints — cauda (origem) e ponta (destino).

O pipeline aqui tem 3 passos:
    1. detect_flows()       — roda o modelo de pose e extrai as setas
    2. match_connections()  — casa cada ponta de seta com o componente
                              detectado mais próximo (Camada 1 / YOLOv8)
    3. analyze_crossings()  — cruza as conexões com as fronteiras de
                              confiança (boundary_*) e sinaliza travessias

É uma feature ADITIVA: alimenta o parâmetro `connections` (reservado desde o
início em stride/analyzer.py) e uma seção própria do relatório, sem alterar
o pipeline de 3 camadas já validado (SLM → Claude/GPT-4o → Consensus).
"""

import math
from pathlib import Path
from typing import Optional

# Pesos do modelo de pose (setas) — separado do best.pt de componentes.
DEFAULT_FLOW_WEIGHTS = Path(__file__).parent / "weights" / "stride_yolo11_pose" / "best.pt"

# O modelo foi treinado com imgsz=1280 (ver README do autor no HF) — rodar em
# outra resolução degrada a detecção. Não confundir com o 640 do modelo de
# componentes: cada modelo infere na resolução em que foi treinado.
FLOW_IMGSZ = 1280
FLOW_CONF = 0.30

BOUNDARY_CLASSES = {
    "boundary_cloud", "boundary_region", "boundary_resource_group",
    "boundary_vpc_or_vnet", "boundary_subnet_public", "boundary_subnet_private",
    "boundary_autoscaling_group",
}

# Rótulos legíveis para as notas de travessia.
BOUNDARY_LABELS = {
    "boundary_cloud": "fronteira da nuvem",
    "boundary_region": "região",
    "boundary_resource_group": "resource group",
    "boundary_vpc_or_vnet": "VPC/VNet",
    "boundary_subnet_public": "subnet pública",
    "boundary_subnet_private": "subnet privada",
    "boundary_autoscaling_group": "autoscaling group",
}

# ---------------------------------------------------------------------------
# Contramedidas por travessia de fronteira — mesmo padrão da base estática de
# stride/analyzer.py, mas keyed por (direção, classe da fronteira). No STRIDE
# clássico, fluxos de dados são sujeitos principalmente a Tampering (adulteração
# em trânsito), Information Disclosure (interceptação) e DoS — as contramedidas
# abaixo refletem isso, especializadas pelo tipo de fronteira cruzada.
# ---------------------------------------------------------------------------
FLOW_COUNTERMEASURES = {
    ("entra", "boundary_vpc_or_vnet"): [
        "TLS 1.2+ (idealmente mTLS) no tráfego que entra na rede virtual",
        "Security Groups/NSG com allowlist por origem — least privilege de rede",
    ],
    ("sai", "boundary_vpc_or_vnet"): [
        "Egress filtering: allowlist de destinos permitidos na saída da VPC/VNet",
        "NAT/proxy de saída com logging para trilha de auditoria",
    ],
    ("entra", "boundary_subnet_public"): [
        "Expor somente as portas/protocolos estritamente necessários",
        "WAF e proteção DDoS no ponto de entrada público",
    ],
    ("sai", "boundary_subnet_public"): [
        "Validar que a subnet pública só encaminha para o tier imediatamente seguinte",
    ],
    ("entra", "boundary_subnet_private"): [
        "Regra de segmentação: apenas o tier anterior pode alcançar a subnet privada",
        "Private endpoints/links em vez de rotas expostas",
    ],
    ("sai", "boundary_subnet_private"): [
        "Controle de egress da subnet privada (rotas e SG/NSG de saída restritos)",
    ],
    ("entra", "boundary_cloud"): [
        "Autenticação forte na borda do provedor (identidade federada, mTLS ou API keys rotativas)",
    ],
    ("sai", "boundary_cloud"): [
        "Criptografia fim a fim para dados que deixam o provedor",
        "DLP/monitoração de exfiltração no tráfego de saída",
    ],
    ("entra", "boundary_resource_group"): [
        "RBAC segregado entre resource groups (isolamento administrativo)",
    ],
    ("sai", "boundary_resource_group"): [
        "RBAC segregado entre resource groups (isolamento administrativo)",
    ],
    ("entra", "boundary_region"): [
        "Criptografia na replicação/tráfego inter-região e atenção a residência de dados",
    ],
    ("sai", "boundary_region"): [
        "Criptografia na replicação/tráfego inter-região e atenção a residência de dados",
    ],
    ("entra", "boundary_autoscaling_group"): [
        "Instâncias novas devem herdar os mesmos controles (golden image/IaC)",
    ],
    ("sai", "boundary_autoscaling_group"): [
        "Instâncias novas devem herdar os mesmos controles (golden image/IaC)",
    ],
}

# Reforço específico do caso de maior interesse (público → privado).
PUBLIC_TO_PRIVATE_COUNTERMEASURES = [
    "Inspeção de tráfego (IDS/IPS) no salto público → privado",
    "Negação por padrão entre subnets, liberando apenas fluxos explícitos",
]

# Fluxos internos (sem travessia de fronteira) também merecem o básico de
# proteção de dados em trânsito — princípio zero trust: a rede interna não é
# automaticamente confiável (STRIDE: T e I se aplicam a qualquer data flow).
INTERNAL_FLOW_COUNTERMEASURES = [
    "TLS também no tráfego interno (zero trust — não confiar na rede interna)",
    "Autenticação serviço-a-serviço (mTLS, service mesh ou identidade gerenciada)",
]

_model_cache: dict = {}


def _load_flow_model(weights_path: Optional[str] = None):
    from ultralytics import YOLO  # import tardio, como em slm/predict.py

    path = str(weights_path or DEFAULT_FLOW_WEIGHTS)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Pesos do modelo de fluxos não encontrados em '{path}'. "
            "Baixe o best.pt de huggingface.co/guillherms/vision-architecture-analyzer-yolo11-pose "
            "para slm/weights/stride_yolo11_pose/."
        )
    if path not in _model_cache:
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


def detect_flows(image_path: str, weights_path: Optional[str] = None,
                 conf: float = FLOW_CONF) -> list[dict]:
    """
    Detecta as setas de fluxo na imagem.

    Returns:
        Lista de dicts: {"tail": (x, y), "head": (x, y), "confidence": float}
        — coordenadas em pixels da imagem original; tail = origem, head = destino.
    """
    model = _load_flow_model(weights_path)
    results = model.predict(source=image_path, imgsz=FLOW_IMGSZ, conf=conf, verbose=False)

    flows = []
    for result in results:
        if result.keypoints is None or result.boxes is None:
            continue
        for box, kps in zip(result.boxes, result.keypoints):
            pts = kps.xy[0].tolist()
            if len(pts) < 2:
                continue
            (tx, ty), (hx, hy) = pts[0], pts[1]
            flows.append({
                "tail": (round(tx, 1), round(ty, 1)),
                "head": (round(hx, 1), round(hy, 1)),
                "confidence": round(float(box.conf[0]), 4),
            })
    return flows


def _match_endpoint(point, components, max_dist, inflate=0.35):
    """
    Casa uma ponta de seta com um componente:
    1º) ponto dentro da bbox inflada em 35% (setas costumam terminar na borda
        do ícone ou no rótulo logo abaixo dele, não no centro);
    2º) senão, componente de centro mais próximo dentro de max_dist.
    Fronteiras (boundary_*) nunca são endpoint — seta liga ativos, não caixas.
    """
    x, y = point
    best, best_dist = None, float("inf")

    for comp in components:
        if comp["class_name"] in BOUNDARY_CLASSES:
            continue
        x1, y1, x2, y2 = comp["bbox"]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.dist((x, y), (cx, cy))
        inside = (x1 - w * inflate <= x <= x2 + w * inflate
                  and y1 - h * inflate <= y <= y2 + h * inflate)
        if inside and dist < best_dist:
            best, best_dist = comp, dist

    if best is not None:
        return best

    for comp in components:
        if comp["class_name"] in BOUNDARY_CLASSES:
            continue
        x1, y1, x2, y2 = comp["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.dist((x, y), (cx, cy))
        if dist < best_dist:
            best, best_dist = comp, dist

    return best if best is not None and best_dist <= max_dist else None


def match_connections(components: list[dict], flows: list[dict],
                      image_size: tuple[int, int]) -> tuple[list[dict], int]:
    """
    Converte setas em conexões componente→componente.

    Args:
        components: saída de slm.predict.detect_components() (bbox em pixels)
        flows:      saída de detect_flows()
        image_size: (largura, altura) da imagem original

    Returns:
        (connections, orphan_count) — connections no formato
        {"from_component": id, "to_component": id, "confidence": float};
        órfãs são setas com alguma ponta sem componente próximo o bastante
        (ex. ícone que o modelo de componentes não detectou).
    """
    diag = math.hypot(*image_size)
    max_dist = 0.09 * diag

    connections, seen = [], set()
    orphans = 0
    for flow in flows:
        src = _match_endpoint(flow["tail"], components, max_dist)
        dst = _match_endpoint(flow["head"], components, max_dist)
        if src is None or dst is None or src["component_id"] == dst["component_id"]:
            orphans += 1
            continue
        key = (src["component_id"], dst["component_id"])
        if key in seen:  # setas duplicadas sobre a mesma linha
            continue
        seen.add(key)
        connections.append({
            "from_component": src["component_id"],
            "to_component": dst["component_id"],
            "confidence": flow["confidence"],
        })
    return connections, orphans


def _boundaries_containing(component, boundaries):
    """Conjunto de fronteiras cuja bbox contém o CENTRO do componente."""
    x1, y1, x2, y2 = component["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    containing = set()
    for b in boundaries:
        bx1, by1, bx2, by2 = b["bbox"]
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            containing.add(b["component_id"])
    return containing


def analyze_crossings(components: list[dict], connections: list[dict]) -> list[dict]:
    """
    Para cada conexão, verifica quais fronteiras de confiança ela cruza
    (fronteiras que contêm uma ponta mas não a outra) e anota o achado.

    Returns:
        Lista de dicts: {"from_component", "to_component", "confidence",
        "crossings": [{"boundary": id, "class_name": str, "direction": "entra"|"sai"}],
        "note": str}
    """
    by_id = {c["component_id"]: c for c in components}
    boundaries = [c for c in components if c["class_name"] in BOUNDARY_CLASSES]

    enriched = []
    for conn in connections:
        src = by_id.get(conn["from_component"])
        dst = by_id.get(conn["to_component"])
        if src is None or dst is None:
            continue
        src_bounds = _boundaries_containing(src, boundaries)
        dst_bounds = _boundaries_containing(dst, boundaries)

        crossings = []
        for b in boundaries:
            bid = b["component_id"]
            if bid in dst_bounds and bid not in src_bounds:
                crossings.append({"boundary": bid, "class_name": b["class_name"], "direction": "entra"})
            elif bid in src_bounds and bid not in dst_bounds:
                crossings.append({"boundary": bid, "class_name": b["class_name"], "direction": "sai"})

        notes = []
        for c in crossings:
            label = BOUNDARY_LABELS.get(c["class_name"], c["class_name"])
            if c["direction"] == "entra":
                notes.append(f"entra na {label} ({c['boundary']}) — validar autenticação/criptografia na entrada")
            else:
                notes.append(f"sai da {label} ({c['boundary']}) — validar controle de egress/destinos permitidos")
        # Caso especial de maior interesse: público → privado
        src_public = any(by_id.get(b, {}).get("class_name") == "boundary_subnet_public" for b in src_bounds)
        dst_private = any(by_id.get(b, {}).get("class_name") == "boundary_subnet_private" for b in dst_bounds)
        if src_public and dst_private:
            notes.insert(0, "fluxo cruza de subnet PÚBLICA para PRIVADA — ponto prioritário de segmentação/inspeção")

        countermeasures = []
        for c in crossings:
            for cm in FLOW_COUNTERMEASURES.get((c["direction"], c["class_name"]), []):
                if cm not in countermeasures:
                    countermeasures.append(cm)
        if src_public and dst_private:
            for cm in PUBLIC_TO_PRIVATE_COUNTERMEASURES:
                if cm not in countermeasures:
                    countermeasures.insert(0, cm)
        if not crossings:
            countermeasures = list(INTERNAL_FLOW_COUNTERMEASURES)

        enriched.append({
            **conn,
            "crossings": crossings,
            "note": "; ".join(notes) if notes else "fluxo interno à mesma fronteira",
            "countermeasures": countermeasures,
        })
    return enriched


def analyze_data_flows(image_path: str, components: list[dict],
                       image_size: tuple[int, int],
                       weights_path: Optional[str] = None) -> dict:
    """
    Função de conveniência — pipeline completo de fluxos para a API:
    detecta setas, casa com componentes e analisa travessias de fronteira.

    Returns:
        {"connections": [...], "crossings": [...], "orphan_flows": int,
         "total_flows": int}
    """
    flows = detect_flows(image_path, weights_path)
    connections, orphans = match_connections(components, flows, image_size)
    crossings = analyze_crossings(components, connections)
    return {
        "total_flows": len(flows),
        "connections": connections,
        "crossings": crossings,
        "orphan_flows": orphans,
    }
