"""
Validação via OpenAI API — GPT-4o (Camada 2 — Validação Multi-LLM).

Mesmo contrato de entrada/saída que claude_validator.py para facilitar o
Consensus Engine (validation/consensus.py).

Requer OPENAI_API_KEY no ambiente (copie .env.example para .env e preencha).

Uso:
    from validation.openai_validator import validate

    result = validate("diagrama.png", [
        {"component_id": "comp_1", "class_name": "data_database", "label": "RDS"},
        {"component_id": "comp_2", "class_name": "edge_waf", "label": "WAF"},
    ])
"""

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from validation.common import (
    SYSTEM_PROMPT,
    encode_image_b64,
    guess_media_type,
    components_to_prompt_json,
    extract_json,
)

load_dotenv()

# Configurável via .env — troque se quiser usar outro modelo com visão (ex. gpt-4o-mini para testes baratos).
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def encode_image(image_path: str) -> str:
    """Mantido por compatibilidade com o contrato original do módulo."""
    return encode_image_b64(image_path)


def validate(image_path: str, slm_components: list) -> dict:
    """
    Envia imagem + componentes detectados para o GPT-4o e retorna análise STRIDE.

    Args:
        image_path:      caminho para o PNG/JPG do diagrama
        slm_components:  lista de dicts com class_name/label/bbox (saída do YOLO)

    Returns:
        dict {component_id: {"S": {...}, ..., "E": {...}}} — mesmo schema do claude_validator

    Raises:
        RuntimeError: se OPENAI_API_KEY não estiver configurada
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não configurada — copie .env.example para .env e preencha."
        )

    client = OpenAI(api_key=api_key)
    media_type = guess_media_type(image_path)
    image_b64 = encode_image_b64(image_path)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},  # força JSON válido do lado da API
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Componentes detectados pelo modelo de visão (YOLO):\n"
                            f"{components_to_prompt_json(slm_components)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                    },
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content
    return extract_json(raw_text)
