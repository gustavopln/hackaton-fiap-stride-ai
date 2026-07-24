"""
Validação via OpenAI API — GPT-4o (Camada 2 — Validação Multi-LLM).

Recebe: imagem do diagrama + JSON com componentes detectados pelo SLM
Retorna: análise STRIDE por componente, mesmo schema do claude_validator.py
(ver validation/common.py — SYSTEM_PROMPT compartilhado entre os dois).

Modelo configurável via env var OPENAI_MODEL (default: gpt-4o).
"""

import json
import os

from openai import OpenAI

from validation.common import (
    SYSTEM_PROMPT,
    components_to_prompt_json,
    encode_image_b64,
    extract_json,
    guess_media_type,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# 4096 truncava o JSON no meio de diagramas com muitos componentes (13+
# componentes x 6 categorias STRIDE facilmente passa de 4096 tokens de saída,
# cortando a resposta e quebrando o parser). gpt-4o suporta até 16384 tokens
# de saída — 8192 dá bastante folga sem chegar perto do limite.
MAX_TOKENS = 8192

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. Defina no .env na raiz do projeto "
                "e reinicie o backend (uvicorn) — variáveis de ambiente só são lidas na subida do processo."
            )
        # timeout explícito — sem isso, uma chamada travada na rede fica pendurada
        # indefinidamente e o Streamlit só mostra "demorou demais", sem log nenhum
        # no terminal do backend (o request nem chegou a falhar do lado do SDK).
        _client = OpenAI(api_key=api_key, timeout=90.0)
    return _client


def _call_openai(image_b64: str, media_type: str, components_json: str, correction: dict = None) -> str:
    """Chama a Chat Completions API (com response_format json_object) com a
    imagem + componentes; opcionalmente reenviando com pedido de correção."""
    client = _get_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Componentes detectados pelo SLM (YOLOv8):\n{components_json}"},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            ],
        },
    ]
    if correction:
        messages.append({"role": "assistant", "content": correction["bad_text"]})
        messages.append({
            "role": "user",
            "content": (
                f"Sua resposta anterior não é um JSON válido ({correction['error']}). "
                "Responda de novo, ESTRITAMENTE em JSON válido, sem markdown, sem texto fora do JSON."
            ),
        })

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
        response_format={"type": "json_object"},
    )
    truncated = response.choices[0].finish_reason == "length"
    return response.choices[0].message.content, truncated


def validate(image_path: str, slm_components: list) -> dict:
    """
    Envia imagem + componentes detectados para o GPT-4o e retorna análise STRIDE.

    Args:
        image_path:      caminho para o PNG/JPG do diagrama
        slm_components:  lista de dicts com class_name e bbox do SLM

    Returns:
        dict com ameaças STRIDE por componente (mesmo schema do claude_validator)

    Raises:
        RuntimeError: se OPENAI_API_KEY não estiver configurada
        json.JSONDecodeError: se, mesmo após 1 retentativa de correção, o
            modelo não retornar JSON válido.
    """
    image_b64 = encode_image_b64(image_path)
    media_type = guess_media_type(image_path)
    components_json = components_to_prompt_json(slm_components)

    text, truncated = _call_openai(image_b64, media_type, components_json)
    try:
        return extract_json(text)
    except json.JSONDecodeError as exc:
        error_msg = (
            "resposta cortada por atingir o limite de tokens — seja mais conciso nas "
            "\"description\" (1 frase curta) para caber todos os componentes"
            if truncated else str(exc)
        )
        print(f"[openai_validator] 1ª resposta não é JSON válido ({error_msg}) — refazendo com correção...")
        text_retry, _ = _call_openai(
            image_b64, media_type, components_json,
            correction={"bad_text": text, "error": error_msg},
        )
        return extract_json(text_retry)
