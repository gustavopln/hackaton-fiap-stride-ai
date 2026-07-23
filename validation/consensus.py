"""
Consensus Engine — Camada 3.

Combina os outputs do SLM (heurística do stride/analyzer.py sobre a detecção
do YOLO), Claude e GPT-4o em um relatório unificado com score de concordância
por ameaça.

Regras:
    - 3/3 concordam  → confidence = "high"
    - 2/3 concordam  → confidence = "medium"
    - 1/3 (só um)    → confidence = "low" (flag para revisão)
    - Desempate de severidade em caso de discordância envolvendo "critical":
      Claude > GPT-4o > SLM (ver ESTRATEGIA_EXECUCAO.md, seção 9 "Riscos e Mitigações").
      Fora desse caso, severidade = média ponderada dos modelos que marcaram a
      ameaça como aplicável.
"""

from typing import Literal, Optional

STRIDE_CATEGORIES = ["S", "T", "R", "I", "D", "E"]

# Ordem usada tanto para desempate de severidade quanto para "detected_by".
SOURCES = ["SLM", "Claude", "OpenAI"]
MODEL_PRIORITY = ["Claude", "OpenAI", "SLM"]  # desempate p/ ameaças críticas

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SEVERITY_BY_RANK = {v: k for k, v in SEVERITY_RANK.items()}


def calculate_agreement(analyses: list) -> Literal["high", "medium", "low"]:
    """
    Calcula o nível de concordância entre as análises dos 3 modelos para uma
    mesma (componente, categoria STRIDE).

    Args:
        analyses: lista de até 3 dicts {"applicable": bool, ...} ou None se o
                  modelo não tiver avaliado esse componente. Ordem esperada:
                  [slm, claude, openai] (mesma ordem de SOURCES).

    Returns:
        "high"   — os 3 modelos concordam entre si se a ameaça é aplicável ou não
        "medium" — 2 de 3 concordam
        "low"    — só 1 avaliou, ou não há maioria clara — flag para revisão manual
    """
    present = [a for a in analyses if a is not None]
    total = len(present)
    if total == 0:
        return "low"

    applicable_votes = sum(1 for a in present if a.get("applicable"))

    if applicable_votes == 0:
        # Nenhum modelo confirma a ameaça nessa categoria — mantém confiança
        # baixa em vez de "alta confiança de que não é ameaça", para não
        # mascarar categorias que nenhum modelo chegou a analisar de verdade.
        return "low"
    if total == 3 and applicable_votes == 3:
        return "high"
    if applicable_votes >= 2:
        return "medium"
    return "low"


def _pick_severity(analyses: list, sources: list) -> str:
    """
    Severidade final para uma (componente, categoria):
    - Se todos os modelos que marcaram "applicable" concordam na severidade, usa ela.
    - Se discordam e a maior severidade envolvida é "critical", usa a hierarquia
      Claude > GPT-4o > SLM (desempate para ameaças críticas).
    - Caso contrário, usa a média (arredondada) dos ranks de severidade.
    """
    applicable = [
        (src, a) for src, a in zip(sources, analyses)
        if a is not None and a.get("applicable")
    ]
    if not applicable:
        return "low"

    ranks = [SEVERITY_RANK.get(a.get("severity", "low"), 1) for _, a in applicable]

    if len(set(ranks)) == 1:
        return SEVERITY_BY_RANK[ranks[0]]

    if max(ranks) == SEVERITY_RANK["critical"]:
        for preferred in MODEL_PRIORITY:
            match = next((a for src, a in applicable if src == preferred), None)
            if match is not None:
                return match.get("severity", "low")

    avg_rank = round(sum(ranks) / len(ranks))
    avg_rank = max(1, min(4, avg_rank))
    return SEVERITY_BY_RANK[avg_rank]


def _merge_descriptions(analyses: list, sources: list) -> str:
    """Concatena as descrições únicas de cada modelo que marcou a ameaça como aplicável."""
    parts = []
    for src, a in zip(sources, analyses):
        if a is None or not a.get("applicable"):
            continue
        desc = (a.get("description") or "").strip()
        if desc and desc not in parts:
            parts.append(f"[{src}] {desc}")
    return " | ".join(parts) if parts else "Sem detalhamento disponível."


def _merge_countermeasures(analyses: list, sources: list) -> list:
    """
    Une as contramedidas de cada modelo que marcou a ameaça como aplicável,
    preservando a ordem (SLM primeiro, depois Claude, depois OpenAI) e
    removendo duplicatas. Hoje só o SLM (stride/analyzer.py, base estática de
    contramedidas) preenche esse campo; Claude/OpenAI ficam prontos para
    contribuir também caso os validators passem a retorná-lo no futuro.
    """
    merged = []
    for _src, a in zip(sources, analyses):
        if a is None or not a.get("applicable"):
            continue
        for c in a.get("countermeasures") or []:
            if c not in merged:
                merged.append(c)
    return merged


def merge_components(slm_output: dict, claude_output: dict, openai_output: dict) -> list:
    """
    Une os component_ids conhecidos pelos 3 modelos em uma lista única,
    preservando a ordem de primeira aparição (SLM, depois Claude, depois OpenAI).
    """
    seen = []
    for source in (slm_output or {}, claude_output or {}, openai_output or {}):
        for component_id in source:
            if component_id not in seen:
                seen.append(component_id)
    return seen


def build_final_report(slm_output: dict, claude_output: dict, openai_output: dict) -> dict:
    """
    Constrói o relatório STRIDE final com scores de confiança.

    Args:
        slm_output:    output do stride/analyzer.py (heurística sobre a detecção do YOLO)
        claude_output: output de validation.claude_validator.validate()
        openai_output: output de validation.openai_validator.validate()

    Returns:
        dict com estrutura:
        {
            "components": [...],
            "threats": {
                "component_id": {
                    "S": {"applicable":..., "description":..., "severity":...,
                          "confidence":..., "detected_by":[...], "countermeasures":[...]},
                    "T": {...}, "R": {...}, "I": {...}, "D": {...}, "E": {...}
                }
            },
            "summary": {
                "total_components": int,
                "agreement_high": int,
                "agreement_medium": int,
                "agreement_low": int,
            }
        }
    """
    outputs = [slm_output or {}, claude_output or {}, openai_output or {}]
    component_ids = merge_components(*outputs)

    threats_by_component = {}
    summary_counts = {"high": 0, "medium": 0, "low": 0}

    for component_id in component_ids:
        threats = {}
        for category in STRIDE_CATEGORIES:
            analyses = [
                (output.get(component_id) or {}).get(category)
                for output in outputs
            ]
            confidence = calculate_agreement(analyses)
            present = [a for a in analyses if a is not None]
            is_applicable = any(a.get("applicable") for a in present) if present else False

            detected_by = [
                src for src, a in zip(SOURCES, analyses)
                if a is not None and a.get("applicable")
            ]

            threats[category] = {
                "applicable": is_applicable,
                "description": _merge_descriptions(analyses, SOURCES),
                "severity": _pick_severity(analyses, SOURCES),
                "confidence": confidence,
                "detected_by": detected_by,
                "countermeasures": _merge_countermeasures(analyses, SOURCES),
            }
            if is_applicable:
                summary_counts[confidence] += 1

        threats_by_component[component_id] = threats

    return {
        "components": component_ids,
        "threats": threats_by_component,
        "summary": {
            "total_components": len(component_ids),
            "agreement_high": summary_counts["high"],
            "agreement_medium": summary_counts["medium"],
            "agreement_low": summary_counts["low"],
        },
    }
