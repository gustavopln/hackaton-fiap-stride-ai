"""
Gerador de diagramas sintéticos de arquitetura cloud com bounding boxes automáticos.

Saída: imagens PNG + labels YOLO (.txt) + labels Florence-2 (.json)

Dependências:
    pip install diagrams pillow

Graphviz (executável) também precisa estar instalado:
    Windows: https://graphviz.org/download/  (adicionar ao PATH)
    Linux:   sudo apt install graphviz
    Mac:     brew install graphviz

Uso:
    # Gerar 200 diagramas com split train/val automático
    python slm/generate_dataset.py --output dataset/synthetic --count 200 --split

    # Gerar sem split (para inspecionar primeiro)
    python slm/generate_dataset.py --output dataset/synthetic --count 50
"""

import os
import re
import json
import random
import argparse
import subprocess
import tempfile
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# 1. MAPEAMENTO DE CLASSES (mesmo do dataset stride-architecture-components-v1)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0:  "actor_user",
    1:  "actor_admin",
    2:  "edge_ddos_protection",
    3:  "edge_cdn",
    4:  "edge_waf",
    5:  "edge_gateway",
    6:  "edge_portal",
    7:  "external_entry_point",
    8:  "integration_orchestrator",
    9:  "integration_messaging",
    10: "compute_load_balancer",
    11: "compute_service",
    12: "compute_worker",
    13: "data_database",
    14: "data_cache",
    15: "data_storage",
    16: "security_identity_provider",
    17: "security_key_management",
    18: "obs_monitoring",
    19: "obs_audit",
    20: "external_backend_service",
    21: "external_saas_service",
    22: "external_web_service",
    23: "communication_service",
    24: "backup_service",
    25: "boundary_cloud",
    26: "boundary_region",
    27: "boundary_resource_group",
    28: "boundary_vpc_or_vnet",
    29: "boundary_subnet_public",
    30: "boundary_subnet_private",
    31: "boundary_autoscaling_group",
}
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

# ---------------------------------------------------------------------------
# 2. MAPEAMENTO: nodes da lib `diagrams` → classes do dataset
# ---------------------------------------------------------------------------
# Cada entrada: (import_path, class_name_in_lib, class_id_no_dataset, label)
AWS_NODES = [
    ("diagrams.aws.general",     "User",            0,  "User"),
    ("diagrams.aws.general",     "User",            1,  "Admin"),
    ("diagrams.aws.network",     "Shield",          2,  "Shield/DDoS"),
    ("diagrams.aws.network",     "CloudFront",      3,  "CloudFront"),
    ("diagrams.aws.network",     "WAF",             4,  "WAF"),
    ("diagrams.aws.network",     "APIGateway",      5,  "API Gateway"),
    ("diagrams.aws.network",     "InternetGateway", 7,  "Internet Gateway"),
    ("diagrams.aws.integration", "StepFunctions",   8,  "Step Functions"),
    ("diagrams.aws.integration", "SQS",             9,  "SQS"),
    ("diagrams.aws.integration", "SNS",             9,  "SNS"),
    ("diagrams.aws.network",     "ELB",             10, "Load Balancer"),
    ("diagrams.aws.network",     "ALB",             10, "ALB"),
    ("diagrams.aws.compute",     "EC2",             11, "EC2"),
    ("diagrams.aws.compute",     "Lambda",          11, "Lambda"),
    ("diagrams.aws.compute",     "ECS",             11, "ECS"),
    ("diagrams.aws.compute",     "EKS",             11, "EKS"),
    ("diagrams.aws.compute",     "Fargate",         12, "Fargate"),
    ("diagrams.aws.database",    "RDS",             13, "RDS"),
    ("diagrams.aws.database",    "Dynamodb",        13, "DynamoDB"),
    ("diagrams.aws.database",    "Aurora",          13, "Aurora"),
    ("diagrams.aws.database",    "ElastiCache",     14, "ElastiCache"),
    ("diagrams.aws.storage",     "S3",              15, "S3"),
    ("diagrams.aws.storage",     "EFS",             15, "EFS"),
    ("diagrams.aws.security",    "Cognito",         16, "Cognito"),
    ("diagrams.aws.security",    "IAM",             16, "IAM"),
    ("diagrams.aws.security",    "KMS",             17, "KMS"),
    ("diagrams.aws.management",  "Cloudwatch",      18, "CloudWatch"),
    ("diagrams.aws.management",  "CloudtrailCloudTrail", 19, "CloudTrail"),
    ("diagrams.aws.network",     "VPC",             28, "VPC"),
    ("diagrams.aws.general",     "General",         20, "External Service"),
    ("diagrams.aws.integration", "Eventbridge",     8,  "EventBridge"),
    ("diagrams.aws.network",     "Route53",         5,  "Route53"),
    ("diagrams.aws.storage",     "Backup",          24, "Backup"),
]

AZURE_NODES = [
    ("diagrams.azure.general",    "Userresource",       0,  "User"),
    ("diagrams.azure.network",    "ApplicationGateway", 10, "App Gateway"),
    ("diagrams.azure.network",    "Frontdoors",         3,  "Front Door"),
    ("diagrams.azure.network",    "Firewall",           4,  "Firewall"),
    ("diagrams.azure.integration","APIManagement",      5,  "API Management"),
    ("diagrams.azure.compute",    "FunctionApps",       11, "Function Apps"),
    ("diagrams.azure.compute",    "AppServices",        11, "App Service"),
    ("diagrams.azure.compute",    "KubernetesServices", 11, "AKS"),
    ("diagrams.azure.database",   "SQLDatabases",       13, "SQL Database"),
    ("diagrams.azure.database",   "CosmosDb",           13, "Cosmos DB"),
    ("diagrams.azure.database",   "CacheForRedis",      14, "Redis Cache"),
    ("diagrams.azure.storage",    "BlobStorage",        15, "Blob Storage"),
    ("diagrams.azure.security",   "ActiveDirectory",    16, "Active Directory"),
    ("diagrams.azure.security",   "KeyVaults",          17, "Key Vault"),
    ("diagrams.azure.monitor",    "Monitor",            18, "Monitor"),
    ("diagrams.azure.network",    "VirtualNetworks",    28, "VNet"),
    ("diagrams.azure.integration","ServiceBus",         9,  "Service Bus"),
    ("diagrams.azure.integration","LogicApps",          8,  "Logic Apps"),
]

GCP_NODES = [
    ("diagrams.gcp.compute",   "GCE",            11, "Compute Engine"),
    ("diagrams.gcp.compute",   "GKE",            11, "GKE"),
    ("diagrams.gcp.compute",   "CloudFunctions", 11, "Cloud Functions"),
    ("diagrams.gcp.compute",   "CloudRun",       11, "Cloud Run"),
    ("diagrams.gcp.database",  "SQL",            13, "Cloud SQL"),
    ("diagrams.gcp.database",  "Spanner",        13, "Spanner"),
    ("diagrams.gcp.database",  "Firestore",      13, "Firestore"),
    ("diagrams.gcp.database",  "Memorystore",    14, "Memorystore"),
    ("diagrams.gcp.storage",   "GCS",            15, "Cloud Storage"),
    ("diagrams.gcp.network",   "LoadBalancing",  10, "Load Balancing"),
    ("diagrams.gcp.network",   "CDN",            3,  "Cloud CDN"),
    ("diagrams.gcp.security",  "IAP",            16, "IAP"),
    ("diagrams.gcp.security",  "KMS",            17, "Cloud KMS"),
    ("diagrams.gcp.operations","Monitoring",     18, "Cloud Monitoring"),
    ("diagrams.gcp.network",   "VPC",            28, "VPC"),
    ("diagrams.gcp.analytics", "PubSub",         9,  "Pub/Sub"),
]

# ---------------------------------------------------------------------------
# 3. TEMPLATES DE ARQUITETURA
# ---------------------------------------------------------------------------
TEMPLATES = [
    {
        "name": "aws_web_3tier",
        "provider": "aws",
        "description": "Web app 3 camadas com WAF e CloudFront",
        "nodes": [
            ("user",  "actor_user",            "User"),
            ("cdn",   "edge_cdn",              "CloudFront"),
            ("waf",   "edge_waf",              "WAF"),
            ("alb",   "compute_load_balancer", "ALB"),
            ("app1",  "compute_service",       "EC2 App 1"),
            ("app2",  "compute_service",       "EC2 App 2"),
            ("db",    "data_database",         "RDS MySQL"),
            ("cache", "data_cache",            "ElastiCache"),
            ("s3",    "data_storage",          "S3"),
            ("iam",   "security_identity_provider", "IAM"),
            ("cw",    "obs_monitoring",        "CloudWatch"),
        ],
        "edges": [
            ("user","cdn"), ("cdn","waf"), ("waf","alb"),
            ("alb","app1"), ("alb","app2"),
            ("app1","db"), ("app2","db"),
            ("app1","cache"), ("app2","cache"),
            ("app1","s3"), ("cw","app1"), ("cw","app2"),
        ],
        "clusters": [("VPC", ["alb","app1","app2","db","cache"])],
    },
    {
        "name": "aws_serverless_api",
        "provider": "aws",
        "description": "API serverless com Lambda e DynamoDB",
        "nodes": [
            ("user",    "actor_user",               "User"),
            ("apigw",   "edge_gateway",             "API Gateway"),
            ("waf",     "edge_waf",                 "WAF"),
            ("lambda1", "compute_service",          "Lambda Auth"),
            ("lambda2", "compute_service",          "Lambda Business"),
            ("lambda3", "compute_service",          "Lambda Data"),
            ("dynamo",  "data_database",            "DynamoDB"),
            ("s3",      "data_storage",             "S3"),
            ("cognito", "security_identity_provider","Cognito"),
            ("kms",     "security_key_management",  "KMS"),
            ("cw",      "obs_monitoring",           "CloudWatch"),
            ("ct",      "obs_audit",                "CloudTrail"),
        ],
        "edges": [
            ("user","apigw"), ("apigw","waf"), ("waf","lambda1"),
            ("lambda1","cognito"), ("lambda1","lambda2"),
            ("lambda2","dynamo"), ("lambda2","lambda3"),
            ("lambda3","s3"), ("kms","dynamo"), ("kms","s3"),
            ("cw","lambda1"), ("cw","lambda2"), ("ct","lambda3"),
        ],
        "clusters": [],
    },
    {
        "name": "aws_microservices_eks",
        "provider": "aws",
        "description": "Microserviços em EKS com service mesh",
        "nodes": [
            ("user",   "actor_user",            "User"),
            ("cf",     "edge_cdn",              "CloudFront"),
            ("alb",    "compute_load_balancer", "ALB"),
            ("svc1",   "compute_service",       "User Service"),
            ("svc2",   "compute_service",       "Order Service"),
            ("svc3",   "compute_service",       "Payment Service"),
            ("worker", "compute_worker",        "Background Worker"),
            ("rds",    "data_database",         "RDS PostgreSQL"),
            ("redis",  "data_cache",            "Redis"),
            ("sqs",    "integration_messaging", "SQS"),
            ("s3",     "data_storage",          "S3"),
            ("cw",     "obs_monitoring",        "CloudWatch"),
        ],
        "edges": [
            ("user","cf"), ("cf","alb"), ("alb","svc1"),
            ("alb","svc2"), ("svc2","svc3"),
            ("svc1","rds"), ("svc2","rds"), ("svc3","rds"),
            ("svc1","redis"), ("svc2","sqs"),
            ("sqs","worker"), ("worker","s3"), ("cw","svc1"),
        ],
        "clusters": [("EKS Cluster", ["svc1","svc2","svc3","worker"])],
    },
    {
        "name": "aws_data_pipeline",
        "provider": "aws",
        "description": "Pipeline de dados com S3 e workers",
        "nodes": [
            ("ext",    "external_entry_point", "IoT / External"),
            ("gw",     "edge_gateway",         "API Gateway"),
            ("lambda", "compute_service",      "Ingest Lambda"),
            ("sqs",    "integration_messaging","SQS"),
            ("worker", "compute_worker",       "ETL Worker"),
            ("s3raw",  "data_storage",         "S3 Raw"),
            ("s3proc", "data_storage",         "S3 Processed"),
            ("rds",    "data_database",        "RDS Analytics"),
            ("backup", "backup_service",       "Backup"),
            ("cw",     "obs_monitoring",       "CloudWatch"),
        ],
        "edges": [
            ("ext","gw"), ("gw","lambda"), ("lambda","sqs"),
            ("sqs","worker"), ("worker","s3raw"),
            ("s3raw","s3proc"), ("s3proc","rds"),
            ("rds","backup"), ("cw","worker"),
        ],
        "clusters": [],
    },
    {
        "name": "aws_multi_region",
        "provider": "aws",
        "description": "Alta disponibilidade multi-region",
        "nodes": [
            ("user",  "actor_user",            "User"),
            ("r53",   "edge_gateway",          "Route 53"),
            ("cf",    "edge_cdn",              "CloudFront"),
            ("waf",   "edge_waf",              "WAF"),
            ("alb1",  "compute_load_balancer", "ALB us-east-1"),
            ("alb2",  "compute_load_balancer", "ALB us-west-2"),
            ("app1",  "compute_service",       "App us-east-1"),
            ("app2",  "compute_service",       "App us-west-2"),
            ("db1",   "data_database",         "RDS Primary"),
            ("db2",   "data_database",         "RDS Replica"),
            ("s3",    "data_storage",          "S3 Cross-Region"),
            ("cw",    "obs_monitoring",        "CloudWatch"),
        ],
        "edges": [
            ("user","r53"), ("r53","cf"), ("cf","waf"),
            ("waf","alb1"), ("waf","alb2"),
            ("alb1","app1"), ("alb2","app2"),
            ("app1","db1"), ("app2","db2"),
            ("db1","db2"), ("app1","s3"), ("app2","s3"),
            ("cw","app1"), ("cw","app2"),
        ],
        "clusters": [
            ("us-east-1", ["alb1","app1","db1"]),
            ("us-west-2", ["alb2","app2","db2"]),
        ],
    },
    {
        "name": "aws_security_focused",
        "provider": "aws",
        "description": "Arquitetura focada em segurança",
        "nodes": [
            ("user",    "actor_user",               "User"),
            ("admin",   "actor_admin",              "Admin"),
            ("shield",  "edge_ddos_protection",     "Shield Advanced"),
            ("waf",     "edge_waf",                 "WAF"),
            ("alb",     "compute_load_balancer",    "ALB"),
            ("app",     "compute_service",          "App Service"),
            ("db",      "data_database",            "RDS Encrypted"),
            ("cognito", "security_identity_provider","Cognito"),
            ("kms",     "security_key_management",  "KMS"),
            ("cw",      "obs_monitoring",           "CloudWatch"),
            ("ct",      "obs_audit",                "CloudTrail"),
            ("iam",     "security_identity_provider","IAM"),
        ],
        "edges": [
            ("user","shield"), ("admin","iam"), ("shield","waf"),
            ("waf","alb"), ("alb","app"), ("app","db"),
            ("cognito","app"), ("kms","db"), ("kms","app"),
            ("cw","app"), ("ct","app"), ("iam","app"),
        ],
        "clusters": [],
    },
    {
        "name": "aws_ecommerce",
        "provider": "aws",
        "description": "E-commerce AWS",
        "nodes": [
            ("user",   "actor_user",            "Customer"),
            ("cf",     "edge_cdn",              "CloudFront"),
            ("waf",    "edge_waf",              "WAF"),
            ("alb",    "compute_load_balancer", "ALB"),
            ("web",    "compute_service",       "Web Server"),
            ("api",    "compute_service",       "API Server"),
            ("worker", "compute_worker",        "Order Worker"),
            ("db",     "data_database",         "RDS Orders"),
            ("dynamo", "data_database",         "DynamoDB Catalog"),
            ("redis",  "data_cache",            "Redis Sessions"),
            ("s3",     "data_storage",          "S3 Assets"),
            ("sqs",    "integration_messaging", "SQS Orders"),
            ("sns",    "communication_service", "SNS Notify"),
            ("cw",     "obs_monitoring",        "CloudWatch"),
        ],
        "edges": [
            ("user","cf"), ("cf","waf"), ("waf","alb"),
            ("alb","web"), ("alb","api"), ("api","worker"),
            ("web","redis"), ("api","redis"),
            ("api","db"), ("api","dynamo"),
            ("worker","sqs"), ("sqs","sns"),
            ("web","s3"), ("cw","web"), ("cw","api"),
        ],
        "clusters": [("VPC Private", ["api","worker","db","dynamo","redis"])],
    },
    {
        "name": "aws_ml_platform",
        "provider": "aws",
        "description": "Plataforma ML na AWS",
        "nodes": [
            ("scientist","actor_admin",              "Data Scientist"),
            ("apigw",    "edge_gateway",             "API Gateway"),
            ("sagemaker","compute_service",          "SageMaker"),
            ("lambda",   "compute_service",          "Inference Lambda"),
            ("worker",   "compute_worker",           "Training Job"),
            ("s3_data",  "data_storage",             "S3 Training Data"),
            ("s3_model", "data_storage",             "S3 Models"),
            ("dynamo",   "data_database",            "DynamoDB Results"),
            ("step",     "integration_orchestrator", "Step Functions"),
            ("cw",       "obs_monitoring",           "CloudWatch"),
            ("ct",       "obs_audit",                "CloudTrail"),
        ],
        "edges": [
            ("scientist","apigw"), ("apigw","step"),
            ("step","worker"), ("worker","s3_data"),
            ("worker","sagemaker"), ("sagemaker","s3_model"),
            ("s3_model","lambda"), ("lambda","dynamo"),
            ("cw","sagemaker"), ("ct","step"),
        ],
        "clusters": [],
    },
    {
        "name": "azure_web_app",
        "provider": "azure",
        "description": "Web app Azure com API Management",
        "nodes": [
            ("user",  "actor_user",                "User"),
            ("fd",    "edge_cdn",                  "Front Door"),
            ("fw",    "edge_waf",                  "Firewall"),
            ("apim",  "edge_gateway",              "API Management"),
            ("app1",  "compute_service",           "App Service"),
            ("app2",  "compute_service",           "Function App"),
            ("sql",   "data_database",             "SQL Database"),
            ("redis", "data_cache",                "Redis Cache"),
            ("blob",  "data_storage",              "Blob Storage"),
            ("ad",    "security_identity_provider","Active Directory"),
            ("kv",    "security_key_management",   "Key Vault"),
            ("mon",   "obs_monitoring",            "Monitor"),
        ],
        "edges": [
            ("user","fd"), ("fd","fw"), ("fw","apim"),
            ("apim","app1"), ("apim","app2"),
            ("app1","sql"), ("app2","sql"),
            ("app1","redis"), ("app1","blob"),
            ("ad","app1"), ("kv","app1"), ("mon","app1"),
        ],
        "clusters": [("Resource Group", ["apim","app1","app2","sql","redis"])],
    },
    {
        "name": "azure_microservices",
        "provider": "azure",
        "description": "Microserviços Azure com AKS",
        "nodes": [
            ("user",   "actor_user",                "User"),
            ("apim",   "edge_gateway",              "API Management"),
            ("svc1",   "compute_service",           "Identity Service"),
            ("svc2",   "compute_service",           "Order Service"),
            ("svc3",   "compute_service",           "Catalog Service"),
            ("sb",     "integration_messaging",     "Service Bus"),
            ("la",     "integration_orchestrator",  "Logic Apps"),
            ("sql",    "data_database",             "SQL Database"),
            ("cosmos", "data_database",             "Cosmos DB"),
            ("ad",     "security_identity_provider","Azure AD"),
            ("mon",    "obs_monitoring",            "Monitor"),
        ],
        "edges": [
            ("user","apim"), ("apim","svc1"),
            ("apim","svc2"), ("apim","svc3"),
            ("svc1","ad"), ("svc2","sb"),
            ("sb","la"), ("la","svc3"),
            ("svc1","sql"), ("svc3","cosmos"),
            ("mon","svc1"), ("mon","svc2"),
        ],
        "clusters": [("AKS", ["svc1","svc2","svc3"])],
    },
    {
        "name": "azure_security_hub",
        "provider": "azure",
        "description": "Hub de segurança Azure",
        "nodes": [
            ("user",  "actor_user",                "User"),
            ("admin", "actor_admin",               "Security Admin"),
            ("fw",    "edge_waf",                  "Azure Firewall"),
            ("apim",  "edge_gateway",              "API Management"),
            ("app",   "compute_service",           "App Service"),
            ("fn",    "compute_service",           "Function App"),
            ("sql",   "data_database",             "SQL Database"),
            ("ad",    "security_identity_provider","Azure AD"),
            ("kv",    "security_key_management",   "Key Vault"),
            ("mon",   "obs_monitoring",            "Monitor"),
            ("blob",  "data_storage",              "Blob Storage"),
            ("la",    "integration_orchestrator",  "Logic Apps"),
        ],
        "edges": [
            ("user","fw"), ("admin","ad"),
            ("fw","apim"), ("apim","app"),
            ("apim","fn"), ("app","sql"),
            ("ad","app"), ("kv","app"), ("kv","sql"),
            ("mon","app"), ("la","fn"), ("fn","blob"),
        ],
        "clusters": [("Resource Group", ["app","fn","sql"])],
    },
    {
        "name": "gcp_web_app",
        "provider": "gcp",
        "description": "Web app GCP com Cloud Run",
        "nodes": [
            ("user", "actor_user",                "User"),
            ("cdn",  "edge_cdn",                  "Cloud CDN"),
            ("lb",   "compute_load_balancer",     "Cloud LB"),
            ("svc1", "compute_service",           "Cloud Run API"),
            ("svc2", "compute_service",           "Cloud Run Auth"),
            ("sql",  "data_database",             "Cloud SQL"),
            ("gcs",  "data_storage",              "Cloud Storage"),
            ("mem",  "data_cache",                "Memorystore"),
            ("iap",  "security_identity_provider","IAP"),
            ("kms",  "security_key_management",   "Cloud KMS"),
            ("mon",  "obs_monitoring",            "Cloud Monitoring"),
        ],
        "edges": [
            ("user","cdn"), ("cdn","lb"), ("lb","svc1"),
            ("svc1","svc2"), ("svc2","iap"),
            ("svc1","sql"), ("svc1","gcs"),
            ("svc1","mem"), ("kms","sql"),
            ("mon","svc1"), ("mon","svc2"),
        ],
        "clusters": [],
    },
    {
        "name": "gcp_data_analytics",
        "provider": "gcp",
        "description": "Pipeline analytics GCP",
        "nodes": [
            ("ext",      "external_entry_point", "External Source"),
            ("ps",       "integration_messaging","Pub/Sub"),
            ("fn",       "compute_service",      "Cloud Function"),
            ("gcs_raw",  "data_storage",         "GCS Raw"),
            ("gcs_proc", "data_storage",         "GCS Processed"),
            ("spanner",  "data_database",        "Spanner"),
            ("worker",   "compute_worker",       "Dataflow"),
            ("backup",   "backup_service",       "Backup"),
            ("mon",      "obs_monitoring",       "Monitoring"),
        ],
        "edges": [
            ("ext","ps"), ("ps","fn"),
            ("fn","gcs_raw"), ("gcs_raw","worker"),
            ("worker","gcs_proc"), ("gcs_proc","spanner"),
            ("spanner","backup"), ("mon","fn"), ("mon","worker"),
        ],
        "clusters": [],
    },
    # -----------------------------------------------------------------
    # Templates extras — cobrem classes sem exemplo real algum nenhum
    # (actor_admin, integration_messaging) e classes sem NENHUM template
    # anterior (edge_portal, external_backend_service, external_saas_service,
    # external_web_service), além de reforçar outras classes raras.
    # -----------------------------------------------------------------
    {
        "name": "aws_partner_ecosystem",
        "provider": "aws",
        "description": "Integração com parceiros externos e portal administrativo",
        "nodes": [
            ("admin",       "actor_admin",               "Ops Admin"),
            ("portal",      "edge_portal",               "Admin Portal"),
            ("ddos",        "edge_ddos_protection",      "Shield Advanced"),
            ("gw",          "edge_gateway",              "API Gateway"),
            ("orchestrator","integration_orchestrator",  "Step Functions"),
            ("queue",       "integration_messaging",     "SQS"),
            ("svc",         "compute_service",           "Lambda"),
            ("worker",      "compute_worker",            "Background Worker"),
            ("ext_backend", "external_backend_service",  "Partner Backend"),
            ("ext_saas",    "external_saas_service",     "Payment SaaS"),
            ("ext_web",     "external_web_service",      "Partner Web API"),
            ("comm",        "communication_service",     "SNS Notify"),
            ("audit",       "obs_audit",                 "CloudTrail"),
            ("backup",      "backup_service",            "Backup"),
        ],
        "edges": [
            ("admin","portal"), ("portal","ddos"), ("ddos","gw"),
            ("gw","orchestrator"), ("orchestrator","queue"), ("queue","worker"),
            ("gw","svc"), ("svc","ext_backend"), ("svc","ext_saas"), ("svc","ext_web"),
            ("svc","comm"), ("audit","svc"), ("backup","queue"),
        ],
        "clusters": [],
    },
    {
        "name": "azure_external_partners",
        "provider": "azure",
        "description": "Portal de cliente com integrações externas via Azure",
        "nodes": [
            ("admin",       "actor_admin",               "IT Admin"),
            ("portal",      "edge_portal",               "Customer Portal"),
            ("ddos",        "edge_ddos_protection",      "DDoS Protection"),
            ("fn",          "compute_service",           "Function App"),
            ("worker",      "compute_worker",            "WebJob Worker"),
            ("sb",          "integration_messaging",     "Service Bus"),
            ("logic",       "integration_orchestrator",  "Logic Apps"),
            ("ext_backend", "external_backend_service",  "Legacy Backend"),
            ("ext_saas",    "external_saas_service",     "CRM SaaS"),
            ("ext_web",     "external_web_service",      "Partner API"),
            ("comm",        "communication_service",     "Notification Hub"),
            ("audit",       "obs_audit",                 "Audit Logs"),
            ("backup",      "backup_service",            "Azure Backup"),
        ],
        "edges": [
            ("admin","portal"), ("portal","ddos"), ("ddos","fn"),
            ("fn","logic"), ("logic","sb"), ("sb","worker"),
            ("fn","ext_backend"), ("fn","ext_saas"), ("fn","ext_web"),
            ("fn","comm"), ("audit","fn"), ("backup","sb"),
        ],
        "clusters": [],
    },
]


# ---------------------------------------------------------------------------
# 4. EXTRAÇÃO DE BOUNDING BOXES VIA GRAPHVIZ PLAIN OUTPUT
# ---------------------------------------------------------------------------

def parse_graphviz_plain(plain_text: str, img_width: int, img_height: int,
                         node_class_map: dict) -> list:
    """
    Converte output do `dot -Tplain` em lista de bounding boxes normalizadas.

    Formato plain do Graphviz:
        graph <scale> <width_in> <height_in>
        node  <name>  <x_in> <y_in> <w_in> <h_in> <label> ...

    Coordenadas em polegadas, origem no canto inferior esquerdo (y invertido).
    """
    annotations = []
    graph_w_pts = graph_h_pts = None

    for line in plain_text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "graph":
            try:
                graph_w_pts = float(parts[2]) * 72
                graph_h_pts = float(parts[3]) * 72
            except (IndexError, ValueError):
                pass

        elif parts[0] == "node" and graph_w_pts and graph_h_pts:
            try:
                node_name = parts[1]
                if node_name not in node_class_map:
                    continue
                class_id = node_class_map[node_name]

                x_pts = float(parts[2]) * 72
                y_pts = float(parts[3]) * 72
                w_pts = float(parts[4]) * 72
                h_pts = float(parts[5]) * 72

                # Inverter eixo Y (Graphviz: 0 embaixo; imagem: 0 em cima)
                x_px = x_pts * (img_width  / graph_w_pts)
                y_px = (graph_h_pts - y_pts) * (img_height / graph_h_pts)
                w_px = w_pts * (img_width  / graph_w_pts)
                h_px = h_pts * (img_height / graph_h_pts)

                # Normalizar para [0, 1]
                x_norm = max(0.0, min(1.0, x_px / img_width))
                y_norm = max(0.0, min(1.0, y_px / img_height))
                w_norm = max(0.001, min(1.0, w_px / img_width))
                h_norm = max(0.001, min(1.0, h_px / img_height))

                annotations.append({
                    "class_id":   class_id,
                    "x_center":   x_norm,
                    "y_center":   y_norm,
                    "width":      w_norm,
                    "height":     h_norm,
                    "node_name":  node_name,
                    "class_name": CLASS_NAMES[class_id],
                })
            except (IndexError, ValueError):
                continue

    return annotations


# ---------------------------------------------------------------------------
# 5. GERAÇÃO DO DIAGRAMA + LABELS
# ---------------------------------------------------------------------------

def _import_node(module_path: str, class_name: str):
    """Importa dinamicamente um node da lib diagrams."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        return None


def _get_catalog(provider: str) -> list:
    if provider == "aws":
        return AWS_NODES
    elif provider == "azure":
        return AZURE_NODES
    elif provider == "gcp":
        return GCP_NODES
    return AWS_NODES


def generate_diagram(template: dict, output_path: Path, diagram_idx: int,
                     randomize: bool = True) -> dict | None:
    """
    Gera um diagrama PNG e retorna anotações YOLO + Florence-2.

    Returns:
        dict com chaves 'image_path', 'yolo_label', 'florence2_label',
        'n_components', 'template', 'provider' — ou None em caso de erro.
    """
    from diagrams import Diagram, Cluster, Edge

    provider  = template["provider"]
    catalog   = _get_catalog(provider)
    fname_base = f"{diagram_idx:04d}_{template['name']}"
    png_path  = output_path / f"{fname_base}.png"
    dot_path  = output_path / f"{fname_base}.dot"

    node_class_map: dict[str, int] = {}

    directions = ["LR", "TB", "RL", "BT"]
    graph_attr = {
        "dpi":     "96",
        "rankdir": random.choice(directions) if randomize else "LR",
        "bgcolor": random.choice(["white", "#f8f8f8", "#ffffff"]) if randomize else "white",
        "pad":     "0.5",
        "nodesep": str(round(random.uniform(0.5, 1.5), 1)) if randomize else "0.8",
        "ranksep": str(round(random.uniform(0.5, 1.5), 1)) if randomize else "0.8",
    }

    try:
        with Diagram(
            template["description"],
            filename=str(png_path.with_suffix("")),
            outformat="png",
            show=False,
            graph_attr=graph_attr,
        ) as diag:
            node_objects: dict = {}

            def create_nodes(node_list: list) -> None:
                for node_id, class_name, label in node_list:
                    class_id = CLASS_IDS.get(class_name, 11)
                    options  = [(mp, cn) for mp, cn, cid, _ in catalog if cid == class_id]
                    if options:
                        mod_path, cls_name = random.choice(options) if randomize else options[0]
                    else:
                        mod_path, cls_name = "diagrams.aws.compute", "EC2"

                    NodeClass = _import_node(mod_path, cls_name)
                    if NodeClass is None:
                        NodeClass = _import_node("diagrams.aws.compute", "EC2")

                    node = NodeClass(label)
                    node_objects[node_id] = node
                    # IMPORTANTE: a lib `diagrams` gera um id interno (hash) para
                    # cada node no grafo Graphviz — o `dot -Tplain` identifica os
                    # nodes por esse hash (node.nodeid), não pelo node_id em
                    # português usado aqui no template ("user", "waf", etc.).
                    # Indexar node_class_map pelo node_id antigo fazia parse_graphviz_plain
                    # nunca encontrar correspondência -> "Sem anotações" em 100% dos casos.
                    node_class_map[node.nodeid] = class_id

            created_in_cluster: set = set()
            for cname, cnodes in template.get("clusters", []):
                with Cluster(cname):
                    create_nodes([n for n in template["nodes"] if n[0] in cnodes])
                    created_in_cluster.update(cnodes)

            create_nodes([n for n in template["nodes"] if n[0] not in created_in_cluster])

            for src, dst in template["edges"]:
                if src in node_objects and dst in node_objects:
                    node_objects[src] >> Edge() >> node_objects[dst]

        # Capturar posições via dot -Tplain
        dot_path.write_text(diag.dot.source, encoding="utf-8")
        result = subprocess.run(
            ["dot", "-Tplain", str(dot_path)],
            capture_output=True, text=True, timeout=30
        )
        dot_path.unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"  [WARN] dot -Tplain falhou: {fname_base}")
            return None

        img = Image.open(png_path)
        img_w, img_h = img.size
        img.close()

        annotations = parse_graphviz_plain(result.stdout, img_w, img_h, node_class_map)
        if not annotations:
            print(f"  [WARN] Sem anotações: {fname_base}")
            return None

        # Label YOLO
        yolo_txt = "".join(
            f"{a['class_id']} {a['x_center']:.6f} {a['y_center']:.6f} "
            f"{a['width']:.6f} {a['height']:.6f}\n"
            for a in annotations
        )
        label_path = output_path / f"{fname_base}.txt"
        label_path.write_text(yolo_txt, encoding="utf-8")

        # Label Florence-2
        florence_items = []
        florence_str   = ""
        for a in annotations:
            x1 = max(0,   int((a["x_center"] - a["width"]  / 2) * 999))
            y1 = max(0,   int((a["y_center"] - a["height"] / 2) * 999))
            x2 = min(999, int((a["x_center"] + a["width"]  / 2) * 999))
            y2 = min(999, int((a["y_center"] + a["height"] / 2) * 999))
            florence_items.append({"class_name": a["class_name"], "bbox_0_999": [x1,y1,x2,y2]})
            florence_str += f"{a['class_name']}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"

        florence_path = output_path / f"{fname_base}.json"
        florence_path.write_text(
            json.dumps({
                "image":      f"{fname_base}.png",
                "task":       "<OD>",
                "output":     florence_str,
                "components": florence_items,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"  [OK] {fname_base}.png  ({len(annotations)} componentes)")
        return {
            "image_path":      str(png_path),
            "yolo_label":      str(label_path),
            "florence2_label": str(florence_path),
            "n_components":    len(annotations),
            "template":        template["name"],
            "provider":        template["provider"],
        }

    except Exception as e:
        print(f"  [ERRO] {fname_base}: {e}")
        return None


# ---------------------------------------------------------------------------
# 6. LOOP PRINCIPAL
# ---------------------------------------------------------------------------

def generate_all(output_dir: str, total: int, seed: int = 42, focus_classes: list | None = None) -> dict:
    random.seed(seed)

    out_path  = Path(output_dir)
    imgs_path = out_path / "images"
    lbls_path = out_path / "labels"
    fl2_path  = out_path / "florence2"
    for p in [imgs_path, lbls_path, fl2_path]:
        p.mkdir(parents=True, exist_ok=True)

    # Se --focus-classes foi passado, só sorteia entre os templates que contêm
    # pelo menos uma dessas classes — maximiza quantos exemplos das classes
    # raras saem por imagem gerada, em vez de sortear entre TODOS os templates
    # (o que reforçaria as classes já abundantes, tipo compute_service).
    pool = TEMPLATES
    if focus_classes:
        focus_set = set(focus_classes)
        pool = [
            t for t in TEMPLATES
            if any(class_name in focus_set for _, class_name, _ in t["nodes"])
        ]
        if not pool:
            raise ValueError(f"Nenhum template contém as classes: {focus_classes}")
        print(f"Filtro --focus-classes ativo: {len(pool)}/{len(TEMPLATES)} templates elegíveis "
              f"({', '.join(sorted(focus_set))})")

    print(f"\n=== Gerando {total} diagramas sintéticos ===")
    print(f"Saída: {out_path.resolve()}\n")

    results, idx = [], 0
    while idx < total:
        template = random.choice(pool)
        result   = generate_diagram(template, imgs_path, idx, randomize=True)
        if result:
            # Mover txt e json para subpastas corretas
            txt_src  = Path(result["yolo_label"])
            json_src = Path(result["florence2_label"])
            txt_dst  = lbls_path / txt_src.name
            json_dst = fl2_path  / json_src.name
            if txt_src  != txt_dst:  txt_src.rename(txt_dst)
            if json_src != json_dst: json_src.rename(json_dst)
            result["yolo_label"]      = str(txt_dst)
            result["florence2_label"] = str(json_dst)
            results.append(result)
            idx += 1

    providers = {}
    for r in results:
        providers[r["provider"]] = providers.get(r["provider"], 0) + 1

    summary = {
        "total_generated": len(results),
        "by_provider":     providers,
        "avg_components":  round(sum(r["n_components"] for r in results) / max(len(results), 1), 1),
        "files":           results,
    }
    (out_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    yaml_lines = ["# Dataset sintético — integrar com stride-architecture-components-v1",
                  f"path: {out_path.resolve()}", "train: images", "val:   images", "", "names:"]
    for cid, cname in sorted(CLASS_NAMES.items()):
        yaml_lines.append(f"  {cid}: {cname}")
    (out_path / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"\n=== Concluído: {len(results)} diagramas | providers: {providers} | "
          f"média {summary['avg_components']} componentes/diagrama ===")
    return summary


# ---------------------------------------------------------------------------
# 7. SPLIT TRAIN / VAL
# ---------------------------------------------------------------------------

def split_dataset(output_dir: str, val_ratio: float = 0.2) -> None:
    import shutil

    out_path = Path(output_dir)
    for split in ["train", "val"]:
        for sub in ["images", "labels", "florence2"]:
            (out_path / split / sub).mkdir(parents=True, exist_ok=True)

    all_imgs = sorted((out_path / "images").glob("*.png"))
    random.shuffle(all_imgs)
    n_val    = max(1, int(len(all_imgs) * val_ratio))
    val_set  = {img.stem for img in all_imgs[:n_val]}

    for img_file in all_imgs:
        split    = "val" if img_file.stem in val_set else "train"
        shutil.copy2(img_file, out_path / split / "images" / img_file.name)
        txt = out_path / "labels"   / f"{img_file.stem}.txt"
        jsn = out_path / "florence2"/ f"{img_file.stem}.json"
        if txt.exists(): shutil.copy2(txt, out_path / split / "labels"    / txt.name)
        if jsn.exists(): shutil.copy2(jsn, out_path / split / "florence2" / jsn.name)

    print(f"Split: train={len(all_imgs)-n_val}  val={n_val}")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera diagramas sintéticos de arquitetura cloud com bounding boxes."
    )
    parser.add_argument("--output",    default="dataset/synthetic",
                        help="Pasta de saída (default: dataset/synthetic)")
    parser.add_argument("--count",     type=int, default=200,
                        help="Número de diagramas (default: 200)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Seed aleatória (default: 42)")
    parser.add_argument("--split",     action="store_true",
                        help="Separar em train/val após gerar")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Proporção de validação (default: 0.2)")
    parser.add_argument("--focus-classes", type=str, default=None,
                        help="Lista de class_names separadas por vírgula (ex.: "
                             "edge_portal,external_saas_service) — só sorteia entre "
                             "templates que contêm pelo menos uma delas.")
    args = parser.parse_args()

    focus_classes = args.focus_classes.split(",") if args.focus_classes else None
    summary = generate_all(args.output, args.count, args.seed, focus_classes=focus_classes)
    if args.split:
        split_dataset(args.output, args.val_ratio)
