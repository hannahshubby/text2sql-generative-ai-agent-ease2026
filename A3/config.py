import os
from pathlib import Path

# Base directory for the A3 Agents
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = Path(r"d:\GitHub\text2sql-generative-ai-agent-new")
DATA_DIR = PROJECT_ROOT / "final" / "data"

# Data paths
TERM_LEXICON_JSON = DATA_DIR / "term_lexicon.json"
TTL_CODE_INDEX_JSON = DATA_DIR / "ttl_code_index.json"
HIERARCHY_JSON = DATA_DIR / "anchor_hierarchy.json"
JOIN_GRAPH_JSON = DATA_DIR / "join_graph.json"
COL_TO_TABLES_JSON = DATA_DIR / "col_to_tables.json"

# Output directory
OUT_DIR = BASE_DIR / "out"

class Config:
    def __init__(self):
        self.out_dir = OUT_DIR
        self.sample_query = ""

CFG = Config()
