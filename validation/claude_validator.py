"""
Validação via Claude API (Camada 2 — Validação Multi-LLM).

Recebe: imagem do diagrama + lista de componentes detectados pelo SLM (YOLO)
Retorna: análise STRIDE por componente — mesmo schema de openai_validator.validate(),
para que o Consensus Engine (validation/consensus.py) possa comparar os dois
sem tratamento especial por modelo.

Requer ANTHROPIC_API_KEY no ambiente (copie .env.example para .env e preencha).

Uso:
    from validation.claude_validator import validate

    result = validate("diagrama.png", [
        {"component_id": "comp_1", "class_name": "data_database", "label": "RDS"},
        {"component_id": "comp_2", "class_name": "edge_waf", "label": "WAF"},
    ])
"""

import os
import json

from dotenv import load_dotenv
import anthropic

from validation.common import (
    SYSTEM_PROMPT,
    encode_image_b64,
    guess_media_type,
    components_to_prompt_json,
    extract_json,
)

load_dotenv()

# Configurável via .env — troque se o nome do modelo mudar (ver console.anthropic.com/docs/models).
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")


def encode_image(image_path: str) -> str:
    """Mantido por compatibilidade com o contrato original do módulo."""
    return encode_image_b64(image_path)


def validate(image_path: str, slm_components: list) -> dict:
    """
    Envia imagem + componentes detectados para o Claude e retorna análise STRIDE.

    Args:
        image_path:      caminho para o PNG/JPG do diagrama
        slm_components:  lista de dicts com class_name/label/bbox (saída do YOLO)

    Returns:
        dict {component_id: {"S": {...}, "T": {...}, "R": {...}, "I": {...}, "D": {...}, "E": {...}}}

    Raises:
        RuntimeError:       se ANTHROPIC_API_KEY não estiver configurada
        json.JSONDecodeError: se nem a resposta original nem a de correção forem JSON válido
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada — copie .env.example para .env e preencha."
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": guess_media_type(image_path),
                "data": encode_image_b64(image_path),
            },
        },
        {
            "type": "text",
            "text": (
                "Componentes detectados pelo modelo de visão (YOLO):\n"
                f"{components_to_prompt_json(slm_components)}"
            ),
        },
    ]

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw_text = "".join(block.text for block in response.content if block.type == "text")

    try:
        return extract_json(raw_text)
    except json.JSONDecodeError:
        # Re-prompt de recuperação — mitigação do risco "LLM produz JSON malformado"
        # (ver ESTRATEGIA_EXECUCAO.md, seção 9 "Riscos e Mitigações").
        fix = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=(
                "Você corrige JSON malformado. Responda SOMENTE com o JSON corrigido, "
                "sem comentários nem markdown."
            ),
            messages=[
                {"role": "user", "content": f"Corrija este texto para ser um JSON válido:\n{raw_text}"}
            ],
        )
        fixed_text = "".join(block.text for block in fix.content if block.type == "text")
        return extract_json(fixed_text)
