"""
API FastAPI — STRIDE Architecture Analyzer.

Endpoints:
    POST /analyze  — recebe imagem, orquestra SLM + validação (Claude/GPT-4o) + consensus
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
from pydantic import BaseModel

from slm.predict import _load_model, detect_components
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
    version="0.2.0",
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
    requisição. Sem isso, quem chamar /analyze primeiro (ex. durante uma
    demo ao vivo) paga o custo de carregar o modelo na hora — em teste local
    isso passou de 60s no cold start; com o warm-up, a 1ª requisição real já
    sai em menos de 1s, igual as seguintes (ver slm/predict.py, _model_cache).
    Se os pesos não existirem ainda, só loga o aviso — a API sobe mesmo assim
    e cada chamada a /analyze vai falhar com 503 até o best.pt ser colocado
    em slm/weights/stride_yolov8s/.
    """
    try:
        _load_model()
    except FileNotFoundError as exc:
        print(f"[startup] aviso: {exc}")


class ReportRequest(BaseModel):
    analysis_result: dict
    diagram_name: str = "Diagrama de Arquitetura"
    trust_boundaries: Optional[list] = None


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.2.0"}


async def _run_validators(image_path: str, components: list[dict]) -> tuple[Optional[dict], Optional[dict]]:
    """
    Chama Claude e GPT-4o em paralelo (Camada 2). Cada validador é síncrono/
    bloqueante (chamada HTTP à API do modelo), então roda em thread separada
    via asyncio.to_thread para não travar o event loop nem serializar as duas
    chamadas.

    Se um validador falhar (chave ausente, erro de rede, rate limit, JSON
    malformado mesmo após a recuperação em claude_validator/openai_validator),
    a falha é isolada: aquela fonte vira None e o Consensus Engine simplesmente
    calcula a concordância com as fontes restantes, em vez de derrubar a
    análise inteira. Falha parcial > falha total nesse pipeline.
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
            # Loga no terminal do uvicorn para diagnóstico — sem isso, a falha
            # vira só um "indisponível" genérico no frontend e fica impossível
            # saber se foi chave ausente, rate limit, JSON malformado, etc.
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
    Recebe uma imagem de diagrama de arquitetura e orquestra as 3 camadas:
    SLM (YOLOv8) → validação Claude + GPT-4o (paralelo) → Consensus Engine.

    Retorna o relatório consolidado (mesmo schema de validation.consensus.build_final_report)
    mais os componentes brutos detectados pelo SLM e as fronteiras de confiança
    identificadas no diagrama.
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
                "warnings": ["Nenhum componente detectado nessa imagem — verifique se é um diagrama de arquitetura válido."],
            }

        slm_output = analyze(components, connections=[])
        trust_boundaries = identify_trust_boundaries(components)

        # Fronteiras de confiança (boundary_*) não são ativos atacáveis — o
        # SLM já as exclui do cálculo STRIDE (ver stride/analyzer.py). Não
        # mandamos elas para os validadores LLM também, senão Claude/GPT-4o
        # podem "inventar" ameaças para uma fronteira que o resto do pipeline
        # trata como contexto, e ela reaparece no relatório final via
        # consensus.merge_components (bug encontrado em teste local).
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

        return {
            "diagram_name": image.filename,
            "components_detected": len(components),
            "raw_components": components,
            "analysis": final_report,
            "trust_boundaries": trust_boundaries,
            "warnings": warnings,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/report")
async def generate_report(payload: ReportRequest, x_api_key: Optional[str] = Header(None)):
    """
    Gera o PDF do relatório STRIDE a partir do resultado retornado por /analyze
    (campo "analysis", e opcionalmente "trust_boundaries" e "diagram_name").
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
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Falha ao gerar PDF: {exc}") from exc

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
