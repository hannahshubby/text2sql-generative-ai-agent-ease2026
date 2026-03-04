# -*- coding: utf-8 -*-
"""
Step 6. SQL Planner Agent (v3 - Sectional Logic Designer)
- Persona: SQL Query Logic Designer
- Task: Decompose the natural language query into specific SQL logical sections.
- Focus: High-precision extraction of filtering, ordering, and paging logic.
"""

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

class SqlPlannerAgent:
    def __init__(self, cfg):
        self.cfg = cfg

    def main(self, cfg, step1_result, step4_result, step5_result):
        user_query = step1_result.get("user_query", "")
        # Step 4 result has confirmed -> terms and codes
        confirmed = step4_result.get("confirmed", {})
        terms = confirmed.get("terms", []) or []
        codes = confirmed.get("codes", []) or []
        col_bindings = step5_result.get("column_bindings", {})
        
        # 1. Available Column & Code Context (Semantic Grounding)
        binding_pool = []
        for t in terms:
            physical = (t.get("physicalName") or "").strip().lower()
            table = col_bindings.get(physical, "")
            if not physical: continue
            
            # Enrich with surface terms and potential code labels for matching
            matching_codes = [c.get("label") for c in codes if c.get("physicalName") == physical]
            binding_pool.append({
                "logical_concept": t.get("originalTerm") or t.get("surface") or "",
                "user_term": t.get("surface", ""),
                "column_physical": physical,
                "table_alias": table,
                "full_expr": f"{table}.{physical}" if table else physical,
                "potential_values": matching_codes
            })

        # 2. Sectional Logic Extraction via LLM
        plan = self.design_sql_logic(user_query, binding_pool)

        # 3. Final Metadata
        plan["meta"] = {
            "agent": "6.sql_planner_agent_v3.1",
            "persona": "SQL_Logic_Designer",
            "timestamp": time.time()
        }
        return plan

    def design_sql_logic(self, query: str, pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError:
            return {"error": "OpenAI SDK missing"}

        key = ""
        client = OpenAI(api_key=key)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")

        system_prompt = (
            "You are a 'SQL Query Logic Designer'. Your role is to translate natural language intent into structured SQL logic sections.\n\n"
            "### INPUT DATA:\n"
            "1. user_intent: The raw question from the user.\n"
            "2. available_columns: A list of columns you MUST use. Includes 'logical_concept', 'user_term', and 'potential_values' (codes found in query).\n\n"
            "### TASK SECTIONS:\n"
            "1. [WHERE_CLAUSE]: Identify filters. \n"
            "   - Pay extreme attention to negations ('~가 아닌', '제외', 'not'). Use NEQ or NOT_IN.\n"
            "   - Match 'potential_values' to the user's intent to choose the right columns.\n"
            "   - Operators: EQ, NEQ, IN, NOT_IN, GT, LT, IS_NULL, IS_NOT_NULL.\n"
            "2. [ORDER_BY_CLAUSE]: Identify sorting columns and direction (ASC/DESC).\n"
            "3. [LIMIT_CLAUSE]: Row count restrictions (e.g., 'top 5').\n"
            "4. [UNIQUENESS_LOGIC]: Deduplication key and policy (KEEP_ONE or EXCLUDE_NON_UNIQUE).\n\n"
            "### CONSTRAINTS:\n"
            "- ONLY use provided 'full_expr'.\n"
            "- Provide 'matched_snippet' for every piece of logic derived from the query.\n"
            "- Output strictly in the specified JSON format."
        )

        user_input = {
            "user_intent": query,
            "available_columns": pool
        }

        # Explicitly defining the JSON structure per SQL section
        json_format = {
            "where_sections": [
                {
                    "column": "table.column",
                    "operator": "NEQ",
                    "values": ["value1", "value2"],
                    "is_negation": True,
                    "matched_snippet": "폐쇄가 아니고"
                }
            ],
            "order_by_section": [
                {"column": "table.column", "direction": "DESC", "matched_snippet": "최근 순으로"}
            ],
            "limit_section": {"count": 10, "matched_snippet": "상위 10개"},
            "dedupe_section": {"key_column": "table.column", "policy": "KEEP_ONE", "matched_snippet": "중복 제거"}
        }

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Design SQL logic for this intent: {json.dumps(user_input, ensure_ascii=False)}. Format: {json.dumps(json_format)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

def main(cfg, step1_result, step4_result, step5_result):
    planner = SqlPlannerAgent(cfg)
    return planner.main(cfg, step1_result, step4_result, step5_result)
