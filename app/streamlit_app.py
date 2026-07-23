"""
Interface Streamlit — STRIDE Architecture Analyzer.

Client web do api/main.py: faz upload do diagrama, chama POST /analyze,
desenha as bounding boxes detectadas pelo SLM sobre a imagem, mostra a tabela
de ameaças STRIDE com score de concordância e origem por fonte (SLM/Claude/
GPT-4o), e permite baixar o relatório final em PDF via POST /report.

Pré-requisito: a API precisa estar rodando em outro terminal:
    uvicorn api.main:app --reload --port 8000

Rodar:
    streamlit run app/streamlit_app.py
"""

import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_BASE_URL_DEFAULT = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # segundos — Claude + GPT-4o em paralelo podem levar um tempo em imagens com muitos componentes

SEVERITY_LABELS = {"critical": "🔴 Critical", "high": "🟠 High", "medium": "🟡 Medium", "low": "🟢 Low"}
CONFIDENCE_LABELS = {"high": "🟢 Alta (3/3)", "medium": "🟡 Média (2/3)", "low": "🔴 Baixa (1/3)"}
BOX_COLOR = "#2b6cb0"

st.set_page_config(page_title="STRIDE Architecture Analyzer", page_icon="🔐", layout="wide")


def _draw_boxes(image: Image.Image, components: list) -> Image.Image:
    """Desenha as bounding boxes detectadas pelo SLM (Camada 1) sobre a imagem original."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=3)
        label = f"{comp['class_name']} {comp['confidence']:.0%}"
        tb = draw.textbbox((x1, y1), label)
        label_h = (tb[3] - tb[1]) + 6
        draw.rectangle([x1, max(0, y1 - label_h), tb[2] - tb[0] + x1 + 6, y1], fill=BOX_COLOR)
        draw.text((x1 + 3, max(0, y1 - label_h + 2)), label, fill="white")
    return img


def _threats_to_rows(analysis: dict) -> list:
    """Achata analysis['threats'] em linhas de tabela, mantendo só ameaças aplicáveis."""
    rows = []
    for component_id in analysis.get("components", []):
        comp_threats = analysis.get("threats", {}).get(component_id, {})
        for category, threat in comp_threats.items():
            if not threat.get("applicable"):
                continue
            rows.append({
                "Componente": component_id,
                "Categoria": category,
                "Descrição": threat.get("description", ""),
                "Severidade": SEVERITY_LABELS.get(threat.get("severity"), threat.get("severity")),
                "Confiança": CONFIDENCE_LABELS.get(threat.get("confidence"), threat.get("confidence")),
                "Detectado por": ", ".join(threat.get("detected_by", [])) or "—",
                "Contramedidas": " • ".join(threat.get("countermeasures", [])) or "—",
            })
    return rows


# ── Sidebar: conexão com a API ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuração")
    api_base_url = st.text_input("URL da API", value=API_BASE_URL_DEFAULT).rstrip("/")
    api_key = st.text_input("X-API-Key (se configurada no .env)", type="password", value=os.getenv("APP_API_KEY", ""))

    if st.button("Testar conexão"):
        try:
            r = requests.get(f"{api_base_url}/health", timeout=5)
            if r.ok:
                st.success(f"API online — v{r.json().get('version', '?')}")
            else:
                st.error(f"API respondeu com status {r.status_code}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Não foi possível conectar: {exc}")

    st.divider()
    st.caption(
        "Pipeline: SLM (YOLOv8) → Claude + GPT-4o (paralelo) → Consensus Engine.\n\n"
        "Suba a API antes de analisar:\n`uvicorn api.main:app --port 8000`"
    )

st.title("🔐 STRIDE Architecture Analyzer")
st.caption("Detecção automática de ameaças em diagramas de arquitetura cloud — FIAP Hackathon Fase 5")

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

uploaded_file = st.file_uploader("Envie o diagrama de arquitetura", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.session_state.uploaded_image = image_bytes

    col_preview, col_action = st.columns([3, 1])
    with col_preview:
        st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)
    with col_action:
        st.write("")
        st.write("")
        analyze_clicked = st.button("🔍 Analisar ameaças STRIDE", type="primary", use_container_width=True)

    if analyze_clicked:
        headers = {"X-API-Key": api_key} if api_key else {}
        with st.spinner("Rodando SLM + Claude + GPT-4o + Consensus Engine... pode levar até 1 minuto."):
            try:
                resp = requests.post(
                    f"{api_base_url}/analyze",
                    files={"image": (uploaded_file.name, image_bytes, uploaded_file.type or "image/png")},
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    st.session_state.analysis_data = resp.json()
                else:
                    st.session_state.analysis_data = None
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except ValueError:
                        detail = resp.text
                    st.error(f"Erro {resp.status_code}: {detail}")
            except requests.exceptions.ConnectionError:
                st.session_state.analysis_data = None
                st.error(
                    f"Não foi possível conectar à API em {api_base_url}. "
                    "Confirme que ela está rodando: `uvicorn api.main:app --port 8000`."
                )
            except requests.exceptions.Timeout:
                st.session_state.analysis_data = None
                st.error("A API demorou demais para responder (timeout). Tente novamente.")

data = st.session_state.analysis_data

if data:
    for w in data.get("warnings", []):
        st.warning(w)

    analysis = data["analysis"]
    summary = analysis.get("summary", {})
    total_threats = sum(
        1 for comp in analysis.get("threats", {}).values() for t in comp.values() if t.get("applicable")
    )

    st.subheader("Resumo da análise")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Componentes analisados", summary.get("total_components", 0))
    m2.metric("Ameaças identificadas", total_threats)
    m3.metric("Alta concordância", summary.get("agreement_high", 0))
    m4.metric("Média concordância", summary.get("agreement_medium", 0))
    m5.metric("Baixa concordância", summary.get("agreement_low", 0))

    if st.session_state.uploaded_image and data.get("raw_components"):
        st.subheader("Componentes detectados (Camada 1 — SLM/YOLOv8)")
        original = Image.open(BytesIO(st.session_state.uploaded_image))
        annotated = _draw_boxes(original, data["raw_components"])
        st.image(annotated, use_container_width=True)

    if data.get("trust_boundaries"):
        with st.expander(f"🧭 Fronteiras de confiança identificadas ({len(data['trust_boundaries'])})"):
            for b in data["trust_boundaries"]:
                st.markdown(f"**{b['node_name']}** ({b['class_name']}) — {b['note']}")

    st.subheader("Ameaças STRIDE por componente")
    rows = _threats_to_rows(analysis)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma ameaça aplicável identificada nesta análise.")

    st.divider()
    st.subheader("📄 Relatório em PDF")
    if st.button("Gerar PDF do relatório"):
        headers = {"X-API-Key": api_key} if api_key else {}
        with st.spinner("Gerando PDF..."):
            try:
                report_resp = requests.post(
                    f"{api_base_url}/report",
                    json={
                        "analysis_result": analysis,
                        "diagram_name": data.get("diagram_name") or "Diagrama de Arquitetura",
                        "trust_boundaries": data.get("trust_boundaries"),
                    },
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                if report_resp.status_code == 200:
                    base_name = (data.get("diagram_name") or "diagrama").rsplit(".", 1)[0]
                    st.download_button(
                        "⬇️ Baixar relatório STRIDE (PDF)",
                        data=report_resp.content,
                        file_name=f"stride_report_{base_name}.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.error(f"Erro ao gerar PDF: {report_resp.status_code} — {report_resp.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Falha na chamada à API: {exc}")
else:
    with st.expander("Como funciona", expanded=True):
        st.markdown("""
        1. Faça upload do diagrama de arquitetura (PNG/JPG/WEBP)
        2. Clique em **Analisar ameaças STRIDE**
        3. O sistema detecta os componentes automaticamente (YOLOv8 fine-tuned)
        4. Claude e GPT-4o validam e enriquecem a análise em paralelo
        5. O Consensus Engine calcula o score de concordância por ameaça
        6. Baixe o relatório completo em PDF
        """)
