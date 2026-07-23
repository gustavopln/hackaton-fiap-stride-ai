"""
Utilitários compartilhados pelos validadores de LLM (Claude e OpenAI).

Camada 2 — Validação Multi-LLM. Mantém os dois validadores com o mesmo
contrato de entrada/saída (mesmo prompt, mesmo parser de JSON) para
simplificar o Consensus Engine (Camada 3), que espera receber outputs
no mesmo schema não importa a origem (SLM, Claude ou OpenAI).
"""

import re
import json
from pathlib import Path

STRIDE_CATEGORIES = ["S", "T", "R", "I", "D", "E"]

SYSTEM_PROMPT = """Você é um especialista em segurança de aplicações e modelagem de ameaças STRIDE.
Você recebe uma imagem de um diagrama de arquitetura de software (AWS, Azure, GCP ou genérico)
e uma lista de componentes já detectados por um modelo de visão computacional (YOLO fine-tuned).

Para CADA componente da lista de entrada, avalie as 6 categorias STRIDE:
  S - Spoofing               (falsificação de identidade)
  T - Tampering              (adulteração de dados em trânsito ou repouso)
  R - Repudiation            (negação de ações, falta de auditoria)
  I - Information Disclosure (exposição de dados sensíveis)
  D - Denial of Service      (interrupção de disponibilidade)
  E - Elevation of Privilege (ganho indevido de permissões)

Use a imagem para entender o contexto real (conexões entre componentes, posição,
fronteiras de confiança/VPC/subnet visíveis no diagrama) — não avalie apenas
pelo nome da classe isoladamente.

Responda ESTRITAMENTE em JSON válido, sem markdown, sem texto antes ou depois do JSON,
no formato exato:

{
  "<component_id>": {
    "S": {"applicable": true, "description": "...", "severity": "medium"},
    "T": {"applicable": false, "description": "...", "severity": "low"},
    "R": {"applicable": true, "description": "...", "severity": "high"},
    "I": {"applicable": true, "description": "...", "severity": "medium"},
    "D": {"applicable": false, "description": "...", "severity": "low"},
    "E": {"applicable": false, "description": "...", "severity": "low"}
  }
}

Regras:
- Use EXATAMENTE os "component_id" fornecidos na lista de entrada como chaves do JSON de saída.
- "severity" deve ser sempre um destes valores: "low", "medium", "high", "critical".
- "applicable" é false quando a categoria STRIDE não é relevante para aquele componente
  neste contexto específico — mesmo assim inclua a chave, com "description" explicando o porquê.
- Não invente componentes que não estão na lista de entrada.
- As 6 categorias (S, T, R, I, D, E) devem estar sempre presentes para cada componente.
"""


def encode_image_b64(image_path: str) -> str:
    """Converte imagem para base64."""
    with open(image_path, "rb") as f:
        import base64
        return base64.standard_b64encode(f.read()).decode("utf-8")


def guess_media_type(image_path: str) -> str:
    """Deduz o media type (ex. image/png) a partir da extensão do arquivo."""
    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"image/{ext or 'png'}"


def components_to_prompt_json(slm_components: list) -> str:
    """
    Serializa a lista de componentes detectados pelo SLM/YOLO para o prompt.

    Aceita dicts com qualquer um destes campos (nessa ordem de prioridade
    para o component_id): "component_id", "node_name", "class_name".
    """
    return json.dumps(
        [
            {
                "component_id": c.get("component_id") or c.get("node_name") or c.get("class_name"),
                "class_name": c.get("class_name"),
                "label": c.get("label", c.get("class_name")),
            }
            for c in slm_components
        ],
        ensure_ascii=False,
        indent=2,
    )


def extract_json(text: str) -> dict:
    """
    Extrai o primeiro bloco JSON válido de uma resposta de LLM, tolerando
    cercas de markdown (```json ... ```) e texto acidental antes/depois.

    Levanta json.JSONDecodeError se não conseguir recuperar um JSON válido —
    quem chamar deve tratar isso com uma nova tentativa (re-prompt de correção)
    se necessário. Essa é a mitigação do risco "SLM/LLM produz JSON malformado"
    listado no ESTRATEGIA_EXECUCAO.md.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start:end + 1])
