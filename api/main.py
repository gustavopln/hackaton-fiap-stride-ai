"""
API FastAPI — STRIDE Architecture Analyzer (+ fluxos de dados).

Endpoints:
    POST /analyze  — recebe imagem, orquestra SLM + fluxos + validação (Claude/GPT-4o) + consensus
    POST /report   — gera PDF do relatório STRIDE a partir do resultado de /analyze

Autenticação: se a variável de ambiente APP_API_KEY estiver definida, os
endpoints /analyze e /report passam a exigir o header "X-API-Key" com o
mesmo valor. Se APP_API_KEY não estiver definida (ex. demo local), a
autenticação fica desativada.

Rodar localmente:
    uvicorn api.main:app --reload --port 8000
"""

import asyncio
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image as PILImage
from pydantic import BaseModel

from slm.predict import _load_model, detect_components
from slm.flows import analyze_data_flows
from stride.analyzer import BOUNDARY_CLASSES, analyze, identify_trust_boundaries
from validation import claude_validator, openai_validator
from validation.consensus import build_final_report
from report.generator import generate_pdf

load_dotenv()

APP_API_KEY = os.getenv("APP_API_KEY")
MAX_IMAGE_SIZE_MB = float(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
REPORT_OUTPUT_DIR = Path(os.getenv("REPORT_OUTPUT_DIR", "./reports/output"))
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

app = FastAPI(
    title="STRIDE Architecture Analyzer",
    description="Detecção automática de ameaças em diagramas de arquitetura cloud.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon/demo — restrinja em produção
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(x_api_key: Optional[str]) -> None:
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key ausente ou inválida.")


@app.on_event("startup")
def _warm_up_slm() -> None:
    """
    Carrega os pesos do YOLOv8 assim que a API sobe, em vez de na primeira
    requisição (ver slm/predict.py, _model_cache). Se os pesos não existirem
    ainda, só loga o aviso — a API sobe mesmo assim.
    """
    try:
        _load_model()
    except FileNotFoundError as exc:
        print(f"[startup] aviso: {exc}")


class ReportRequest(BaseModel):
    analysis_result: dict
    diagram_name: str = "Diagrama de Arquitetura"
    trust_boundaries: Optional[list] = None
    flows: Optional[dict] = None


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.3.0"}


async def _run_validators(image_path: str, components: list[dict]) -> tuple[Optional[dict], Optional[dict]]:
    """
    Chama Claude e GPT-4o em paralelo (Camada 2), com isolamento de falha —
    ver comentários detalhados no projeto principal.
    """
    async def _safe_call(label, fn, *args):
        start = time.monotonic()
        print(f"[validators] {label} iniciado...")
        try:
            result = await asyncio.to_thread(fn, *args)
            elapsed = time.monotonic() - start
            print(f"[validators] {label} respondeu em {elapsed:.1f}s")
            return result
        except Exception as exc:  # noqa: BLE001 — isolamento de falha é intencional aqui
            elapsed = time.monotonic() - start
            print(f"[validators] {label} falhou após {elapsed:.1f}s: {exc!r}")
            return {"__error__": str(exc)}

    claude_result, openai_result = await asyncio.gather(
        _safe_call("Claude", claude_validator.validate, image_path, components),
        _safe_call("GPT-4o", openai_validator.validate, image_path, components),
    )

    claude_output = None if isinstance(claude_result, dict) and "__error__" in claude_result else claude_result
    openai_output = None if isinstance(openai_result, dict) and "__error__" in openai_result else openai_result

    return claude_output, openai_output


@app.post("/analyze")
async def analyze_diagram(image: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    """
    Orquestra: SLM (YOLOv8) → fluxos (YOLO11-pose) → validação Claude + GPT-4o
    (paralelo) → Consensus Engine.

    A detecção de fluxos é ADITIVA e tolerante a falha: se os pesos do modelo
    de pose não estiverem presentes (ou a detecção falhar), a análise STRIDE
    segue normalmente, só sem a seção de fluxos — mesma filosofia de
    degradação parcial usada nos validadores LLM.
    """
    _check_api_key(x_api_key)

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: {image.content_type}. Use PNG, JPEG ou WEBP.",
        )

    suffix = Path(image.filename or "diagram.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Imagem maior que o limite de {MAX_IMAGE_SIZE_MB}MB.",
            )
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Camada 1 — SLM (YOLOv8 fine-tuned)
        try:
            components = detect_components(tmp_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if not components:
            return {
                "diagram_name": image.filename,
                "components_detected": 0,
                "analysis": {"components": [], "threats": {}, "summary": {
                    "total_components": 0, "agreement_high": 0, "agreement_medium": 0, "agreement_low": 0,
                }},
                "trust_boundaries": [],
                "flows": None,
                "warnings": ["Nenhum componente detectado nessa imagem — verifique se é um diagrama de arquitetura válido."],
            }

        trust_boundaries = identify_trust_boundaries(components)

        # Camada 1.5 — fluxos de dados (YOLO11-pose, aditivo/tolerante a falha)
        flows_data = None
        flows_warning = None
        try:
            with PILImage.open(tmp_path) as im:
                image_size = im.size
            flows_data = analyze_data_flows(tmp_path, components, image_size)
            print(f"[flows] {flows_data['total_flows']} setas, "
                  f"{len(flows_data['connections'])} conexões, "
                  f"{flows_data['orphan_flows']} órfãs")
            if not flows_data["connections"]:
                # Sem aviso explícito, a seção de fluxos some do relatório em
                # silêncio e o usuário fica sem saber se é bug ou limitação —
                # setas muito finas/pontilhadas ficam fora do alcance do modelo
                # de pose (limitação conhecida, ver README).
                flows_warning = (
                    "Nenhum fluxo de dados mapeado nesta imagem "
                    f"({flows_data['total_flows']} seta(s) detectada(s), nenhuma com correspondência "
                    "entre componentes) — o relatório segue sem a seção de fluxos. Setas muito finas "
                    "ou pontilhadas estão fora do alcance do detector atual."
                )
        except FileNotFoundError as exc:
            flows_warning = f"Detecção de fluxos indisponível: {exc}"
            print(f"[flows] {flows_warning}")
        except Exception as exc:  # noqa: BLE001 — fluxos nunca derrubam a análise
            flows_warning = "Detecção de fluxos falhou nesta análise — relatório segue sem a seção de fluxos."
            print(f"[flows] falhou: {exc!r}")

        connections = (flows_data or {}).get("connections", [])
        slm_output = analyze(components, connections=connections)

        # Fronteiras de confiança não vão para os validadores LLM (ver projeto principal).
        attackable_components = [c for c in components if c["class_name"] not in BOUNDARY_CLASSES]

        # Camada 2 — Claude + GPT-4o (paralelo, com isolamento de falha)
        claude_output, openai_output = await _run_validators(tmp_path, attackable_components)

        # Camada 3 — Consensus Engine
        final_report = build_final_report(slm_output, claude_output, openai_output)

        warnings = []
        if claude_output is None:
            warnings.append("Validação Claude indisponível nesta análise — confiança calculada só com SLM/GPT-4o.")
        if openai_output is None:
            warnings.append("Validação GPT-4o indisponível nesta análise — confiança calculada só com SLM/Claude.")
        if flows_warning:
            warnings.append(flows_warning)

        return {
            "diagram_name": image.filename,
            "components_detected": len(components),
            "raw_components": components,
            "analysis": final_report,
            "trust_boundaries": trust_boundaries,
            "flows": flows_data,
            "warnings": warnings,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/report")
async def generate_report(payload: ReportRequest, x_api_key: Optional[str] = Header(None)):
    """
    Gera o PDF do relatório STRIDE a partir do resultado retornado por /analyze
    (campos "analysis", e opcionalmente "trust_boundaries", "flows" e "diagram_name").
    """
    _check_api_key(x_api_key)

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_OUTPUT_DIR / f"stride_report_{uuid.uuid4().hex[:8]}.pdf"

    try:
        pdf_path = generate_pdf(
            payload.analysis_result,
            str(output_path),
            diagram_name=payload.diagram_name,
            trust_boundaries=payload.trust_boundaries,
            flows=payload.flows,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Falha ao gerar PDF: {exc}") from exc

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
