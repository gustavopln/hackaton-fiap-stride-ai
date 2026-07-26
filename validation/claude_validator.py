"""
Validação via Claude API (Camada 2 — Validação Multi-LLM).

Recebe: imagem do diagrama + JSON com componentes detectados pelo SLM
Retorna: análise STRIDE por componente, no mesmo schema do openai_validator.py
(ver validation/common.py — SYSTEM_PROMPT compartilhado entre os dois).

Modelo configurável via env var CLAUDE_MODEL (default: claude-sonnet-5) —
caso a API rejeite o modelo (ex. "model not found" por descontinuação),
ajuste no .env sem precisar mexer no código.
"""

import json
import os

import anthropic

from validation.common import (
    SYSTEM_PROMPT,
    components_to_prompt_json,
    encode_image_b64,
    extract_json,
    guess_media_type,
)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
# Histórico desse limite: 4096 truncava com 13+ componentes; 8192 resolveu os
# diagramas médios, mas o teste com a arquitetura AWS da FIAP (21 componentes
# atacáveis x 6 categorias = 126 objetos de ameaça no JSON) estourou de novo —
# truncamento de VOLUME DE RESPOSTA, não de thinking (que já está desligado).
# 16384 cobre ~40 componentes com folga. max_tokens é um TETO, não custo
# pré-pago: o cenário caro é o truncamento, que joga fora a 1ª chamada inteira
# e paga uma retentativa completa. Claude Sonnet 5 suporta até 128k de saída.
MAX_TOKENS = 16384

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY não configurada. Defina no .env na raiz do projeto "
                "e reinicie o backend (uvicorn) — variáveis de ambiente só são lidas na subida do processo."
            )
        # timeout explícito — sem isso, uma chamada travada na rede fica pendurada
        # indefinidamente e o Streamlit só mostra "demorou demais", sem log nenhum
        # no terminal do backend (o request nem chegou a falhar do lado do SDK).
        # 150s (era 90s): com MAX_TOKENS=16384, uma resposta grande pode
        # legitimamente passar de 90s de geração — timeout curto demais viraria
        # uma nova fonte de falha justamente nos diagramas maiores.
        _client = anthropic.Anthropic(api_key=api_key, timeout=150.0)
    return _client


def _call_claude(image_b64: str, media_type: str, components_json: str, correction: dict = None) -> str:
    """Chama a Messages API com a imagem + componentes; opcionalmente reenviando
    com um pedido de correção quando a resposta anterior não veio em JSON válido."""
    client = _get_client()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": f"Componentes detectados pelo SLM (YOLOv8):\n{components_json}"},
            ],
        }
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

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        # Claude Sonnet 5 vem com "adaptive thinking" LIGADO por padrão
        # (effort=high) — os tokens de raciocínio interno contam dentro do
        # mesmo max_tokens e são cobrados, competindo com o JSON de resposta.
        # É a causa real do 1º JSON vir cortado e do consumo na Anthropic
        # estar mais alto que na OpenAI, não o modelo em si. Nossa tarefa é
        # extração estruturada bem especificada pelo SYSTEM_PROMPT — não
        # precisa de raciocínio estendido, então desligamos.
        thinking={"type": "disabled"},
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    truncated = response.stop_reason == "max_tokens"
    return text, truncated


def validate(image_path: str, slm_components: list) -> dict:
    """
    Envia imagem + componentes detectados para o Claude e retorna análise STRIDE.

    Args:
        image_path:      caminho para o PNG/JPG do diagrama
        slm_components:  lista de dicts com class_name e bbox do SLM

    Returns:
        dict com ameaças STRIDE por componente (mesmo schema do openai_validator)

    Raises:
        RuntimeError: se ANTHROPIC_API_KEY não estiver configurada
        json.JSONDecodeError: se, mesmo após 1 retentativa de correção, o
            modelo não retornar JSON válido (mitigação parcial do risco
            "LLM produz JSON malformado" — ver ESTRATEGIA_EXECUCAO.md)
    """
    image_b64 = encode_image_b64(image_path)
    media_type = guess_media_type(image_path)
    components_json = components_to_prompt_json(slm_components)

    text, truncated = _call_claude(image_b64, media_type, components_json)
    try:
        return extract_json(text)
    except json.JSONDecodeError as exc:
        error_msg = (
            "resposta cortada por atingir o limite de tokens — seja mais conciso nas "
            "\"description\" (1 frase curta) para caber todos os componentes"
            if truncated else str(exc)
        )
        print(f"[claude_validator] 1ª resposta não é JSON válido ({error_msg}) — refazendo com correção...")
        text_retry, _ = _call_claude(
            image_b64, media_type, components_json,
            correction={"bad_text": text, "error": error_msg},
        )
        return extract_json(text_retry)
