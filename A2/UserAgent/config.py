import os
from pathlib import Path

# Base directory for the A2 UserAgent
BASE_DIR = Path(r"d:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A2\UserAgent")

# Data paths
CACHE_DIR = BASE_DIR / "cache_terms"
CSV_DIR = BASE_DIR / "csv"
TTL_DIR = BASE_DIR / "ttl"
INTERIM_DIR = BASE_DIR / "interim_result"

# Specific file paths
TERM_COLLECT_CSV = CSV_DIR / "term_collect.csv"
CODE_COLLECT_CSV = CSV_DIR / "code_collect.csv"
ONTOLOGY_TTL = TTL_DIR / "financial_terms.ttl" # Or whichever is the main one
COL_TO_TABLES_JSON = TTL_DIR / "col_to_tables.json"
JOIN_GRAPH_JSON = TTL_DIR / "join_graph.json"
CODEBOOK_JSON = TTL_DIR / "codebook_from_code_collect.json"

# Ensure directories exist
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (Should ideally be in env, but keeping here for consistency with original scripts if needed)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
