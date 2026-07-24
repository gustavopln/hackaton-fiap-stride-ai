# STRIDE Architecture Analyzer

Detecção automática de ameaças de segurança em diagramas de arquitetura cloud (AWS, Azure, GCP), usando modelagem de ameaças STRIDE. Projeto desenvolvido para o Hackathon da Fase 5 (Privacidade, Segurança de Dados e Aplicações Práticas) da pós-tech FIAP.

Você faz upload de um diagrama de arquitetura (PNG/JPG) e o sistema devolve, em poucos minutos, um relatório completo em PDF com as ameaças STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege) aplicáveis a cada componente identificado, a severidade, o nível de concordância entre as fontes de análise, e contramedidas recomendadas.

## Como funciona — 3 camadas

```
Imagem do diagrama
        │
        ▼
┌───────────────────────────┐
│ Camada 1 — SLM (YOLOv8)   │  detecção de componentes (32 classes) +
│  stride/analyzer.py       │  heurística STRIDE + base de contramedidas
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Camada 2 — Validação LLM  │  Claude e GPT-4o analisam a MESMA imagem +
│  (Claude ∥ GPT-4o)        │  componentes detectados, em paralelo e de
│                            │  forma independente uma da outra
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Camada 3 — Consensus       │  combina as 3 análises (SLM + Claude +
│  Engine                    │  GPT-4o), calcula o score de concordância
│  validation/consensus.py   │  (alta/média/baixa) e a severidade final
└───────────────────────────┘
        │
        ▼
  Relatório STRIDE em PDF
```

As 3 fontes analisam o diagrama **de forma independente** — o SLM não valida a saída do Claude/GPT-4o via RAG nem nada parecido; cada uma chega à sua própria conclusão sobre os mesmos componentes, e é só na Camada 3 que os resultados são comparados e combinados. Quando as 3 concordam, a confiança da ameaça é "alta"; quando só uma fonte identifica algo que as outras não viram, a confiança fica "baixa" e fica sinalizado para revisão manual — isso é uma característica desejada do desenho, não um defeito: é o próprio motivo de usar 3 avaliadores independentes em vez de um só.

## Estrutura do projeto

```
hackathon-stride-ai/
├── api/                    # Backend FastAPI (orquestra as 3 camadas)
│   └── main.py             #   POST /analyze, POST /report, GET /health
├── app/
│   └── streamlit_app.py    # Frontend web (client HTTP da API)
├── slm/                    # Camada 1 — detecção de componentes (YOLOv8)
│   ├── generate_dataset.py #   gera o dataset sintético de treino
│   ├── predict.py          #   inferência do modelo treinado
│   ├── weights/             #   pesos do modelo (best.pt)
│   └── notebook-*.ipynb     #   notebook de treino executado no Kaggle
├── stride/
│   └── analyzer.py          # Heurística STRIDE + base de contramedidas
├── validation/               # Camada 2 (Claude/GPT-4o) + Camada 3 (consenso)
│   ├── claude_validator.py
│   ├── openai_validator.py
│   ├── consensus.py
│   └── common.py             #   prompt e utilitários compartilhados
├── report/                   # Geração do relatório em PDF
│   ├── generator.py
│   └── templates/stride_report.html
├── docs/
│   ├── training_results/     # Métricas e gráficos do treino do YOLO
│   └── fiap_test_architectures/  # Diagramas usados no teste end-to-end
├── dataset/synthetic/         # Dataset sintético gerado para o treino
└── requirements.txt
```

## Como rodar localmente

Pré-requisitos: Python 3.11+, chaves de API da Anthropic (Claude) e da OpenAI (GPT-4o).

No Windows, a geração de PDF (WeasyPrint) precisa do runtime GTK3 instalado — veja [aqui](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
# edite o .env e preencha ANTHROPIC_API_KEY e OPENAI_API_KEY
```

Suba o backend e o frontend em dois terminais separados:

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — interface web
streamlit run app/streamlit_app.py
```

Acesse `http://localhost:8501`, faça upload de um diagrama de arquitetura e clique em **Analisar ameaças STRIDE**.

## Camada 1 — treinamento do SLM (YOLOv8)

O modelo de detecção (YOLOv8s) foi treinado do zero para reconhecer **32 classes** de componentes de arquitetura cloud (atores, serviços de borda, computação, dados, segurança, observabilidade, integrações externas e fronteiras de confiança como VPC/subnet/autoscaling group), a partir de um dataset sintético gerado com a biblioteca `diagrams` (`slm/generate_dataset.py`) e complementado com imagens adicionais para cobrir classes raras.

Treino executado no Kaggle (GPU T4 x2) — [notebook completo com os outputs reais](https://www.kaggle.com/code/gustavopln/notebook-stride-architecture-components?scriptVersionId=337310807), também versionado em `slm/notebook-stride-architecture-components.ipynb`. Parada antecipada (early stopping, patience=20) na época 63 de 100, com o melhor checkpoint na época 43:

| Métrica | Valor |
|---|---|
| mAP50 | 0,70 |
| mAP50-95 | 0,58 |
| Precision | 0,55 |
| Recall | 0,88 |

O recall alto (0,88) mostra que o modelo raramente deixa de detectar um componente presente no diagrama — importante para não deixar ameaças de fora da análise. A precision mais moderada (0,55) reflete alguns falsos positivos entre classes visualmente parecidas (ex. variações de ícones de computação); esse ruído é absorvido pelas Camadas 2 e 3, já que Claude e GPT-4o reavaliam cada componente contra a imagem original antes do relatório final. Gráficos completos (matriz de confusão, curvas P/R/F1, batches de validação) em `docs/training_results/`.

A taxonomia das 32 classes e imagens de apoio para desenvolvimento e testes tiveram como referência o dataset público de Guilherme Santos no Hugging Face (ver [Referências](#referências)).

## Teste end-to-end

O pipeline foi validado com as duas arquiteturas de referência fornecidas no enunciado do hackathon (AWS multi-AZ e Azure API Management, em `docs/fiap_test_architectures/`), gerando relatórios completos em PDF com detecção de componentes, ameaças por categoria STRIDE, score de concordância entre as 3 fontes e contramedidas recomendadas para cada ameaça aplicável.

## Decisões de arquitetura e pivots

- **RAG → base de conhecimento estática**: o plano original usava LangChain + ChromaDB para recuperar contramedidas de uma base de documentos (OWASP, MITRE ATT&CK, CWE). Foi substituído por uma base estática em `stride/analyzer.py` (contramedidas genéricas por categoria STRIDE + específicas por classe de componente), suficiente para o escopo do hackathon e sem dependência de infraestrutura extra de indexação/embeddings.
- **Florence-2 → YOLOv8**: a detecção de componentes migrou de um modelo de visão-linguagem (Florence-2) para um YOLOv8 fine-tuned, mais leve e rápido para rodar em CPU na demo, com melhor controle sobre as classes de interesse.

## Stack

Python · FastAPI · Streamlit · Ultralytics YOLOv8 · Anthropic Claude · OpenAI GPT-4o · Jinja2 + WeasyPrint (PDF) · Pillow/diagrams (geração do dataset sintético)

## Referências

- SANTOS, Guilherme. **STRIDE Architecture Threat Modeling Dataset (AWS & Azure)**. Hugging Face Datasets, licença MIT. Disponível em: [huggingface.co/datasets/guillherms/stride-architecture-components-v1](https://huggingface.co/datasets/guillherms/stride-architecture-components-v1). Dataset com 4.190 imagens anotadas em 32 classes (formato YOLO) de componentes de diagramas de arquitetura AWS e Azure — usado como referência para a taxonomia de classes e para imagens de apoio no desenvolvimento e testes deste projeto.

## Autor

Gustavo — [github.com/gustavopln/hackaton-fiap-stride-ai](https://github.com/gustavopln/hackaton-fiap-stride-ai)
