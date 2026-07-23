"""
Orquestrador da análise STRIDE.

Recebe os componentes detectados (Camada 1 — SLM/YOLOv8) + dados de fluxo e
aplica as 6 categorias STRIDE com apoio de uma base de conhecimento estática
de contramedidas (pivot RAG → static KB, ver PLANO_SPRINT_FINAL.md).

STRIDE:
    S — Spoofing        (falsificação de identidade)
    T — Tampering       (adulteração de dados)
    R — Repudiation     (negação de ações)
    I — Info Disclosure (vazamento de informação)
    D — Denial of Service
    E — Elevation of Privilege

A saída de analyze() é o "slm_output" consumido por validation/consensus.py
(build_final_report). O schema por categoria é:
    {"applicable": bool, "description": str, "severity": str, "countermeasures": [str, ...]}
(o campo "confidence" e "detected_by" finais são recalculados pelo Consensus
Engine a partir das 3 fontes — SLM, Claude, OpenAI — e não devem ser lidos
daqui como valor definitivo.)
"""

STRIDE_CATEGORIES = ["S", "T", "R", "I", "D", "E"]

# ---------------------------------------------------------------------------
# 1) Aplicabilidade heurística: classe de componente → categorias STRIDE
#    tipicamente relevantes para esse tipo de asset.
# ---------------------------------------------------------------------------
STRIDE_HEURISTICS = {
    "actor_user": ["S", "R"],
    "actor_admin": ["S", "R", "E"],
    "edge_ddos_protection": ["D"],
    "edge_cdn": ["I", "T"],
    "edge_waf": ["S", "T"],
    "edge_gateway": ["S", "T", "D"],
    "edge_portal": ["S", "T", "I"],
    "external_entry_point": ["S", "T", "I"],
    "integration_orchestrator": ["T", "R", "E"],
    "integration_messaging": ["T", "I", "D"],
    "compute_load_balancer": ["D", "T"],
    "compute_service": ["S", "T", "I", "D", "E"],
    "compute_worker": ["T", "E"],
    "data_database": ["T", "I", "D"],
    "data_cache": ["I", "T"],
    "data_storage": ["I", "T"],
    "security_identity_provider": ["S", "E"],
    "security_key_management": ["T", "I", "E"],
    "obs_monitoring": ["R", "I"],
    "obs_audit": ["R"],
    "external_backend_service": ["S", "T", "I"],
    "external_saas_service": ["I", "T"],
    "external_web_service": ["S", "I"],
    "communication_service": ["I", "T"],
    "backup_service": ["I", "T"],
}

# Classes que representam limites de confiança (trust boundaries) no diagrama,
# não ativos atacáveis — não recebem ameaças STRIDE diretas (ver
# identify_trust_boundaries() abaixo).
BOUNDARY_CLASSES = {
    "boundary_cloud",
    "boundary_region",
    "boundary_resource_group",
    "boundary_vpc_or_vnet",
    "boundary_subnet_public",
    "boundary_subnet_private",
    "boundary_autoscaling_group",
}

BOUNDARY_NOTES = {
    "boundary_cloud": "Limite entre ambiente on-premises/cliente e o provedor de nuvem — ponto de entrada para revisão de shared responsibility model.",
    "boundary_region": "Cruzamento entre regiões — atenção a residência/soberania de dados e latência de replicação.",
    "boundary_resource_group": "Agrupamento administrativo — verificar isolamento de IAM/policies entre grupos.",
    "boundary_vpc_or_vnet": "Fronteira de rede virtual — validar peering, NSG/Security Groups e rotas entre VPCs/VNets.",
    "boundary_subnet_public": "Sub-rede com exposição direta à internet — superfície de ataque prioritária.",
    "boundary_subnet_private": "Sub-rede sem rota direta à internet — validar que não há exposição acidental (NAT mal configurado, etc.).",
    "boundary_autoscaling_group": "Conjunto de instâncias elásticas — cada nova instância deve herdar os mesmos controles de segurança (golden image/IaC).",
}

STRIDE_DESCRIPTIONS = {
    "S": "Spoofing — falsificação de identidade ou autenticação",
    "T": "Tampering — adulteração de dados em trânsito ou em repouso",
    "R": "Repudiation — negação de ações realizadas, falta de auditoria",
    "I": "Information Disclosure — exposição de dados sensíveis",
    "D": "Denial of Service — interrupção da disponibilidade",
    "E": "Elevation of Privilege — ganho indevido de permissões",
}

# ---------------------------------------------------------------------------
# 2) Severidade: baseline por aplicabilidade + overrides para combinações
#    (classe, categoria) de impacto reconhecidamente alto/crítico.
# ---------------------------------------------------------------------------
SEVERITY_OVERRIDES = {
    ("security_identity_provider", "S"): "critical",  # IdP comprometido = bypass total de autenticação
    ("security_key_management", "I"): "critical",      # vazamento de chave = compromete tudo que ela protege
    ("security_key_management", "T"): "high",
    ("data_database", "I"): "high",
    ("data_database", "T"): "high",
    ("actor_admin", "E"): "high",
    ("compute_service", "E"): "high",
    ("edge_waf", "T"): "high",                          # bypass do WAF
    ("integration_orchestrator", "E"): "high",
    ("backup_service", "T"): "high",                    # adulterar backup compromete o último recurso de recuperação
}


def _severity_for(class_name: str, category: str, is_applicable: bool) -> str:
    if not is_applicable:
        return "low"
    return SEVERITY_OVERRIDES.get((class_name, category), "medium")


# ---------------------------------------------------------------------------
# 3) Base de contramedidas — camada genérica (por categoria STRIDE) +
#    camada específica (por classe de componente), somadas na saída final.
#    Isto substitui o pipeline de RAG original (ver PLANO_SPRINT_FINAL.md,
#    seção de pivots) por uma base de conhecimento estática, suficiente para
#    o escopo do hackathon e sem dependência de infraestrutura extra.
# ---------------------------------------------------------------------------
GENERIC_COUNTERMEASURES = {
    "S": [
        "Autenticação forte (MFA) e gestão centralizada de identidade",
        "Validação de certificados/tokens em toda comunicação",
        "Princípio de menor privilégio nas credenciais de serviço",
    ],
    "T": [
        "Assinatura e verificação de integridade dos dados (HMAC/checksum)",
        "TLS obrigatório em trânsito e criptografia em repouso",
        "Validação e sanitização de entradas",
    ],
    "R": [
        "Logging centralizado e imutável (WORM) com timestamp confiável",
        "Assinatura digital de transações críticas",
        "Trilha de auditoria correlacionável por identidade",
    ],
    "I": [
        "Criptografia em repouso e em trânsito",
        "Controle de acesso baseado em papéis (RBAC) e mascaramento de dados sensíveis",
        "Classificação de dados e DLP (Data Loss Prevention)",
    ],
    "D": [
        "Rate limiting e throttling",
        "Auto-scaling e redundância multi-AZ",
        "Proteção contra DDoS na borda (ex.: AWS Shield, Azure DDoS Protection)",
    ],
    "E": [
        "Least privilege e segregação de funções",
        "Revisão periódica de permissões (IAM Access Review)",
        "Isolamento de processos/contêineres (sandboxing)",
    ],
}

COMPONENT_COUNTERMEASURES = {
    ("actor_user", "S"): ["MFA obrigatório para usuários finais", "Detecção de anomalias de login (geolocalização, dispositivo)"],
    ("actor_user", "R"): ["Registro de ações do usuário com correlação de sessão"],

    ("actor_admin", "S"): ["MFA + acesso privilegiado via bastion/PAM (Privileged Access Management)"],
    ("actor_admin", "R"): ["Gravação de sessão administrativa (session recording)"],
    ("actor_admin", "E"): ["Just-in-time (JIT) privilege elevation, sem contas admin permanentes"],

    ("edge_ddos_protection", "D"): ["Configurar limites de taxa por IP/região", "Scrubbing center e absorção de tráfego volumétrico"],

    ("edge_cdn", "I"): ["Assinar URLs (signed URLs/cookies) para conteúdo sensível"],
    ("edge_cdn", "T"): ["Validar integridade de objetos em cache (ETag/hash)"],

    ("edge_waf", "S"): ["Regras contra credential stuffing e bot management"],
    ("edge_waf", "T"): ["Regras OWASP Core Rule Set contra injeção/XSS"],

    ("edge_gateway", "S"): ["Autenticação de API (OAuth2/JWT) no gateway"],
    ("edge_gateway", "T"): ["Validação de schema de payload (OpenAPI/JSON Schema)"],
    ("edge_gateway", "D"): ["Quotas por client_id e circuit breaker"],

    ("edge_portal", "S"): ["Autenticação federada (SSO) e proteção contra session hijacking"],
    ("edge_portal", "T"): ["Content Security Policy (CSP) e proteção anti-CSRF"],
    ("edge_portal", "I"): ["Evitar exposição de dados internos em mensagens de erro"],

    ("external_entry_point", "S"): ["mTLS ou API keys para parceiros externos"],
    ("external_entry_point", "T"): ["Validação estrita de payload na fronteira de confiança"],
    ("external_entry_point", "I"): ["Minimização de dados (data minimization) para consumidores externos"],

    ("integration_orchestrator", "T"): ["Idempotência e validação de mensagens entre etapas do workflow"],
    ("integration_orchestrator", "R"): ["Log de cada transição de estado do workflow com trace ID"],
    ("integration_orchestrator", "E"): ["Segregação de permissões entre etapas (cada step com role própria)"],

    ("integration_messaging", "T"): ["Assinatura de mensagens e verificação de schema no consumidor"],
    ("integration_messaging", "I"): ["Criptografia de payload de fila/tópico"],
    ("integration_messaging", "D"): ["Dead-letter queue e limite de profundidade de fila"],

    ("compute_load_balancer", "D"): ["Health checks agressivos e failover automático"],
    ("compute_load_balancer", "T"): ["TLS termination segura, sem downgrade para HTTP interno desprotegido"],

    ("compute_service", "S"): ["Identidade gerenciada (managed identity/IAM role), sem credenciais estáticas"],
    ("compute_service", "T"): ["Verificação de integridade de artefatos de deploy (imagem assinada)"],
    ("compute_service", "I"): ["Segredos via secret manager, nunca em variável de ambiente em texto claro"],
    ("compute_service", "D"): ["Auto-scaling horizontal com limites de CPU/memória"],
    ("compute_service", "E"): ["Execução com usuário não-root e permissões mínimas de IAM role"],

    ("compute_worker", "T"): ["Validação de mensagens antes do processamento (evitar deserialização insegura)"],
    ("compute_worker", "E"): ["Isolamento de execução (contêiner com privilégios mínimos)"],

    ("data_database", "T"): ["Prepared statements/ORM parametrizado contra SQL Injection"],
    ("data_database", "I"): ["Criptografia em repouso (TDE) e mascaramento de colunas sensíveis"],
    ("data_database", "D"): ["Connection pooling com limites e read replicas para absorver picos"],

    ("data_cache", "I"): ["Não armazenar dados sensíveis em cache sem criptografia/TTL curto"],
    ("data_cache", "T"): ["Autenticação no cache (ex.: Redis AUTH/ACL) contra escrita não autorizada"],

    ("data_storage", "I"): ["Bucket/blob privado por padrão, com política de acesso explícita"],
    ("data_storage", "T"): ["Versionamento e object lock contra sobrescrita maliciosa"],

    ("security_identity_provider", "S"): ["Proteção contra brute-force e detecção de impossible travel"],
    ("security_identity_provider", "E"): ["Revisão de escopos/claims de token (least privilege)"],

    ("security_key_management", "T"): ["Rotação automática de chaves e versionamento"],
    ("security_key_management", "I"): ["HSM ou KMS gerenciado — chaves nunca expostas em texto claro"],
    ("security_key_management", "E"): ["Políticas de uso de chave restritas por role (key policy)"],

    ("obs_monitoring", "R"): ["Retenção mínima de logs conforme compliance, com integridade verificável"],
    ("obs_monitoring", "I"): ["Redação de dados sensíveis (PII) antes de persistir logs/métricas"],

    ("obs_audit", "R"): ["Armazenamento imutável (WORM) e replicação para conta separada de auditoria"],

    ("external_backend_service", "S"): ["Autenticação mútua (mTLS) ou API key rotativa com o parceiro"],
    ("external_backend_service", "T"): ["Contrato de API versionado com validação de schema"],
    ("external_backend_service", "I"): ["Acordo de tratamento de dados (DPA) e filtragem de campos sensíveis"],

    ("external_saas_service", "I"): ["Revisão de escopos de integração (OAuth scopes mínimos)"],
    ("external_saas_service", "T"): ["Validação de assinatura de webhooks (HMAC) recebidos do SaaS"],

    ("external_web_service", "S"): ["Validar certificado TLS do serviço externo (pinning quando crítico)"],
    ("external_web_service", "I"): ["Timeout e circuit breaker para evitar exposição por respostas lentas/verbosas"],

    ("communication_service", "I"): ["Criptografia de mensagens/e-mails com dados sensíveis"],
    ("communication_service", "T"): ["Verificação de remetente (SPF/DKIM/DMARC) contra spoofing de conteúdo"],

    ("backup_service", "I"): ["Criptografia de backups e controle de acesso restrito à restauração"],
    ("backup_service", "T"): ["Verificação de integridade do backup (checksum) antes da restauração"],
}


def _countermeasures_for(class_name: str, category: str) -> list[str]:
    return GENERIC_COUNTERMEASURES[category] + COMPONENT_COUNTERMEASURES.get((class_name, category), [])


def identify_trust_boundaries(components: list[dict]) -> list[dict]:
    """
    Extrai os componentes de fronteira de confiança (boundary_*) detectados
    no diagrama, com uma nota explicativa — usado pelo report/generator.py
    (Task 12) para contextualizar o relatório, não entra no cálculo STRIDE
    por componente (fronteiras não são ativos atacáveis).
    """
    boundaries = []
    for comp in components:
        class_name = comp.get("class_name")
        if class_name in BOUNDARY_CLASSES:
            boundaries.append({
                "node_name": comp.get("node_name", class_name),
                "class_name": class_name,
                "note": BOUNDARY_NOTES.get(class_name, ""),
            })
    return boundaries


def analyze(components: list[dict], connections: list[dict]) -> dict:
    """
    Aplica heurísticas STRIDE + base de contramedidas aos componentes detectados.

    Args:
        components:  lista de dicts com class_name, bbox, label
        connections: lista de dicts com from_component, to_component, protocol
                      (não usado no cálculo ainda — reservado para correlação
                      de fluxo de dados com fronteiras de confiança, Task 15)

    Returns:
        dict {component_name: {categoria_STRIDE: {applicable, description,
        severity, countermeasures}}} — compatível com validation/consensus.py.
        Componentes do tipo boundary_* são excluídos (ver identify_trust_boundaries).
    """
    report = {}
    for comp in components:
        class_name = comp.get("class_name", "compute_service")
        if class_name in BOUNDARY_CLASSES:
            continue  # fronteiras de confiança não recebem ameaças STRIDE diretas

        applicable_categories = STRIDE_HEURISTICS.get(class_name, ["S", "T", "I", "D"])
        threats = {}
        for category in STRIDE_CATEGORIES:
            is_applicable = category in applicable_categories
            threats[category] = {
                "applicable": is_applicable,
                "description": STRIDE_DESCRIPTIONS[category],
                "severity": _severity_for(class_name, category, is_applicable),
                "countermeasures": _countermeasures_for(class_name, category) if is_applicable else [],
            }
        report[comp.get("node_name", class_name)] = threats

    return report
