"""
Gerador de relatório STRIDE em PDF.

Usa Jinja2 para renderizar HTML (report/templates/stride_report.html) e
WeasyPrint para converter o HTML renderizado em PDF.

Entrada esperada (analysis_result): saída de validation/consensus.py
build_final_report() —
    {
        "components": [component_id, ...],
        "threats": {
            component_id: {
                "S": {"applicable": bool, "description": str, "severity": str,
                      "confidence": str, "detected_by": [str, ...],
                      "countermeasures": [str, ...]},
                "T": {...}, "R": {...}, "I": {...}, "D": {...}, "E": {...}
            }
        },
        "summary": {"total_components": int, "agreement_high": int,
                    "agreement_medium": int, "agreement_low": int}
    }
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "stride_report.html"

STRIDE_ORDER = ["S", "T", "R", "I", "D", "E"]
STRIDE_LABELS = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information Disclosure",
    "D": "Denial of Service",
    "E": "Elevation of Privilege",
}
SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)


def _build_component_rows(analysis_result: dict) -> list[dict]:
    """
    Achata threats[component][categoria] em linhas prontas para o template,
    mantendo apenas ameaças "applicable" e ordenando por severidade (mais
    crítico primeiro) dentro de cada componente.
    """
    threats = analysis_result.get("threats", {})
    rows = []
    for component_id in analysis_result.get("components", []):
        component_threats = threats.get(component_id, {})
        applicable = []
        for category in STRIDE_ORDER:
            threat = component_threats.get(category)
            if not threat or not threat.get("applicable"):
                continue
            applicable.append({
                "category": category,
                "category_label": STRIDE_LABELS[category],
                "description": threat.get("description", ""),
                "severity": threat.get("severity", "low"),
                "confidence": threat.get("confidence", "low"),
                "detected_by": threat.get("detected_by", []),
                "countermeasures": threat.get("countermeasures", []),
            })
        applicable.sort(key=lambda t: SEVERITY_ORDER.get(t["severity"], 0), reverse=True)
        rows.append({
            "component_id": component_id,
            "threat_count": len(applicable),
            "threats": applicable,
        })

    # Componentes com ameaças mais numerosas/graves aparecem primeiro no relatório.
    def _rank(row):
        max_sev = max((SEVERITY_ORDER.get(t["severity"], 0) for t in row["threats"]), default=0)
        return (max_sev, row["threat_count"])

    rows.sort(key=_rank, reverse=True)
    return rows


def generate_html(
    analysis_result: dict,
    diagram_name: str = "Diagrama de Arquitetura",
    trust_boundaries: Optional[list] = None,
) -> str:
    """
    Renderiza o relatório STRIDE em HTML. Usado internamente por generate_pdf,
    e também exposto para servir o relatório direto (ex.: preview no Streamlit)
    sem precisar gerar PDF.

    Args:
        analysis_result:   saída do Consensus Engine.
        diagram_name:       nome/identificação do diagrama, exibido no cabeçalho.
        trust_boundaries:   saída opcional de stride.analyzer.identify_trust_boundaries(),
                            exibida como contexto (fronteiras de confiança detectadas).
    """
    template = _env.get_template(TEMPLATE_NAME)
    component_rows = _build_component_rows(analysis_result)

    summary = analysis_result.get("summary") or {
        "total_components": len(analysis_result.get("components", [])),
        "agreement_high": 0,
        "agreement_medium": 0,
        "agreement_low": 0,
    }

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for row in component_rows:
        for t in row["threats"]:
            severity_counts[t["severity"]] = severity_counts.get(t["severity"], 0) + 1
    total_threats = sum(row["threat_count"] for row in component_rows)

    return template.render(
        diagram_name=diagram_name,
        generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
        summary=summary,
        total_threats=total_threats,
        severity_counts=severity_counts,
        components=component_rows,
        trust_boundaries=trust_boundaries or [],
    )


def generate_pdf(
    analysis_result: dict,
    output_path: str,
    diagram_name: str = "Diagrama de Arquitetura",
    trust_boundaries: Optional[list] = None,
) -> Path:
    """
    Gera relatório STRIDE em PDF.

    Args:
        analysis_result:   output do consensus engine (validation/consensus.py
                            build_final_report).
        output_path:        caminho do PDF a ser gerado.
        diagram_name:        nome/identificação do diagrama analisado.
        trust_boundaries:    ver generate_html().

    Returns:
        Path do PDF gerado.
    """
    # Import tardio: só exige weasyprint (+ libs de sistema Pango/Cairo/GDK-Pixbuf)
    # no momento em que um PDF de fato precisa ser gerado.
    from weasyprint import HTML

    html_content = generate_html(
        analysis_result, diagram_name=diagram_name, trust_boundaries=trust_boundaries
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_content, base_url=str(TEMPLATE_DIR)).write_pdf(str(output))
    return output
