# -*- coding: utf-8 -*-
"""
Agent3 (v3.3): Evidence-based Rule Constraint Planner
Principle (근거 기반)
- 사용자 요구/문장에 근거(evidence)가 없으면 어떤 constraint도 "임의로" 생성하지 않는다.
  (limit/order_by/time_window/filters/dedupe 모두 동일)
- Rule 기반으로만 생성한다.
- LLM은 검증 전용이며 기본 OFF.

입력
- temp_artifacts/UserCanonicalRewrite.txt
- temp_artifacts/7_table_join_plan.json (Agent2 output)

출력
- temp_artifacts/8_constraint_plan.json
"""

from __future__ import annotations
import os, re, json
from typing import Any, Dict, List, Optional, Tuple

USE_LLM_VALIDATION = False
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENT_TXT = ""
JOINPLAN_JSON = ""
OUT_JSON = ""

def norm(x: Any) -> str:
    return "" if x is None else str(x).strip()

def fold(s: str) -> str:
    s = norm(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def read_text_robust(path: str) -> Tuple[str, str]:
    if not os.path.exists(path):
        return "", "missing"
    raw = open(path, "rb").read()
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc).strip(), enc
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace").strip(), "utf-8-replace"

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_binding_index(selected_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Indices for binding resolution.
    """
    bindings = selected_plan.get("column_bindings") or []
    pool = []
    for b in bindings:
        concept = norm(b.get("concept"))
        table = norm(b.get("boundTable"))
        col = norm(b.get("column"))
        if not col:
            continue
        expr = f"{table}.{col}" if table else col
        pool.append({"expr": expr, "concept": concept, "table": table, "column": col})
    return {"pool": pool}

def find_target_expr(index: Dict[str, Any], *, concept_keywords: List[str] = None, colname_keywords: List[str] = None) -> Optional[str]:
    pool = index["pool"]
    # strict: all keywords
    if concept_keywords:
        for p in pool:
            c = p.get("concept") or ""
            if c and all(k in c for k in concept_keywords):
                return p["expr"]
    if colname_keywords:
        for p in pool:
            cn = (p.get("column") or "").lower()
            if cn and all(k in cn for k in colname_keywords):
                return p["expr"]

    # relaxed: any keyword
    if concept_keywords:
        for p in pool:
            c = p.get("concept") or ""
            if c and any(k in c for k in concept_keywords):
                return p["expr"]
    if colname_keywords:
        for p in pool:
            cn = (p.get("column") or "").lower()
            if cn and any(k in cn for k in colname_keywords):
                return p["expr"]
    return None

def find_span(text: str, snippet: str) -> Optional[List[int]]:
    if not text or not snippet:
        return None
    i = text.find(snippet)
    if i >= 0:
        return [i, i + len(snippet)]
    return None

def extract_quoted_values(segment: str) -> List[str]:
    return re.findall(r"'([^']+)'", segment)

def parse_limit(intent: str) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """
    Evidence-based: only when explicit.
    """
    s = fold(intent)
    m = re.search(r"(top|상위)\s*(\d+)", s)
    if m:
        n = int(m.group(2))
        snippet = m.group(0)
        return n, {"matched_text": snippet, "rule": "LIMIT_TOP_N", "span": find_span(s, snippet)}
    m = re.search(r"(\d+)\s*(건|개)\s*(만|까지|제한)", s)
    if m:
        n = int(m.group(1))
        snippet = m.group(0)
        return n, {"matched_text": snippet, "rule": "LIMIT_N_ROWS", "span": find_span(s, snippet)}
    return None, None

def parse_order_by(intent: str, pool: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Evidence-based: only if intent explicitly indicates ordering.
    """
    s = fold(intent)
    if not any(k in s for k in ["최근", "최신", "recent", "latest", "newest", "정렬", "오래된", "oldest"]):
        return [], None

    # choose time-like candidate if exists
    time_candidates = []
    for p in pool:
        e = (p.get("expr") or "").lower()
        if any(k in e for k in ["_dt", "date", "time", "_ym", "yyyymm", "created", "updated"]):
            time_candidates.append(p["expr"])
    if not time_candidates:
        return [], {"matched_text": "order signal present but no time-like column", "rule": "ORDER_SIGNAL_NO_TIME_COL", "span": None}

    if any(k in s for k in ["오래된", "oldest"]):
        return [{"expr": time_candidates[0], "dir": "ASC"}], {"matched_text": "오래된/oldest", "rule": "ORDER_OLDEST", "span": None}
    return [{"expr": time_candidates[0], "dir": "DESC"}], {"matched_text": "최근/최신/latest", "rule": "ORDER_RECENT", "span": None}

def extract_rules(intent: str, index: Dict[str, Any], enc: str) -> Dict[str, Any]:
    filters: List[Dict[str, Any]] = []
    dedupe_policy = None

    # Resolve targets from bindings (evidence chain: bound columns from Agent2)
    acct_stat = find_target_expr(index, concept_keywords=["계좌", "상태"], colname_keywords=["stat"]) \
                or find_target_expr(index, colname_keywords=["ac_stat"]) \
                or find_target_expr(index, colname_keywords=["stat", "tcd"])  # AC_STAT_TCD

    rlnm = find_target_expr(index, concept_keywords=["실명", "가명"], colname_keywords=["rlnm"]) \
           or find_target_expr(index, colname_keywords=["rlnm", "tcd"])  # RLNM_NRNM_TCD

    cif = find_target_expr(index, concept_keywords=["고객", "식별"], colname_keywords=["cif"]) \
          or find_target_expr(index, colname_keywords=["cif"])

    user_id = find_target_expr(index, concept_keywords=["온라인", "사용자"], colname_keywords=["onln"]) \
              or find_target_expr(index, colname_keywords=["onln", "user", "id"]) \
              or find_target_expr(index, colname_keywords=["user", "id"])

    # segment by markers
    seg1 = intent
    seg2 = ""
    if "(2)" in intent:
        seg1 = intent.split("(2)")[0]
        tail = intent.split("(2)", 1)[1]
        if "(3)" in tail:
            seg2 = tail.split("(3)")[0]
        else:
            seg2 = tail

    # (1) NOT IN from quoted values + negation language
    if acct_stat:
        vals = extract_quoted_values(seg1)
        if vals and ("아닌" in seg1 or "제외" in seg1 or "not" in fold(seg1)):
            matched = seg1.strip()
            filters.append({
                "target": acct_stat,
                "op": "NOT_IN",
                "value": vals,
                "valueType": "LIST",
                "confidence": 0.9,
                "rationale": "계좌 상태 제외 조건(룰 기반)",
                "evidence": {
                    "source": os.path.basename(INTENT_TXT),
                    "matched_text": matched[:200],
                    "rule": "NOT_IN_FROM_QUOTES_WITH_NEGATION",
                    "span": find_span(intent, seg1.strip())  # best-effort
                }
            })

    # (2) NEQ from quoted value + negation language
    if rlnm:
        vals = extract_quoted_values(seg2) if seg2 else []
        if vals and ("아닌" in seg2 or "제외" in seg2 or "not" in fold(seg2)):
            matched = seg2.strip()
            filters.append({
                "target": rlnm,
                "op": "NEQ",
                "value": vals[0],
                "valueType": "STRING",
                "confidence": 0.9,
                "rationale": "실명/가명 구분코드 제외(룰 기반)",
                "evidence": {
                    "source": os.path.basename(INTENT_TXT),
                    "matched_text": matched[:200],
                    "rule": "NEQ_FROM_QUOTE_WITH_NEGATION",
                    "span": find_span(intent, seg2.strip())
                }
            })

    # (3) CIF 존재(확인되는/존재)
    if cif and any(k in intent for k in ["확인", "존재"]) or ("not null" in fold(intent)):
        snippet = "고객식별번호" if "고객식별번호" in intent else "확인"
        filters.append({
            "target": cif,
            "op": "IS_NOT_NULL",
            "value": None,
            "valueType": "NULL",
            "confidence": 0.75,
            "rationale": "고객식별번호 존재 조건(룰 기반)",
            "evidence": {
                "source": os.path.basename(INTENT_TXT),
                "matched_text": snippet,
                "rule": "IS_NOT_NULL_FROM_EXISTENCE_LANGUAGE",
                "span": find_span(intent, snippet)
            }
        })

    # (4) Dedupe (evidence-based; only if explicit)
    if user_id:
        # Exclude non-unique keys ONLY if explicit "모두 조회하지 않는다/전부 제외"
        if ("중복" in intent) and any(k in intent for k in ["모두 조회하지", "전부", "모두 제외", "조회하지 않는다", "제외한다"]):
            matched = "중복이면 모두 조회하지 않는다"
            span = find_span(intent, matched)
            dedupe_policy = {
                "type": "EXCLUDE_NON_UNIQUE_KEYS",
                "key": user_id,
                "sql_hint": f"Exclude keys that appear more than once: GROUP BY {user_id} HAVING COUNT(*)=1",
                "evidence": {
                    "source": os.path.basename(INTENT_TXT or "in_memory"),
                    "matched_text": matched,
                    "rule": "EXCLUDE_NON_UNIQUE_KEYS_EXPLICIT",
                    "span": span
                }
            }
        # Keep-one policy ONLY if explicit
        elif ("중복" in intent) and any(k in intent for k in ["하나만", "대표", "1건", "중복 제거", "최신 1건"]):
            matched = "중복 제거(대표 1건)"
            dedupe_policy = {
                "type": "DEDUP_KEEP_ONE",
                "key": user_id,
                "order_by": [],
                "sql_hint": f"Keep one row per key using ROW_NUMBER() OVER (PARTITION BY {user_id} ORDER BY <time_col> DESC)=1",
                "evidence": {
                    "source": os.path.basename(INTENT_TXT or "in_memory"),
                    "matched_text": matched,
                    "rule": "DEDUP_KEEP_ONE_EXPLICIT",
                    "span": None
                }
            }

    # (5) General Equality Mapper (Evidence-based)
    # 문장에서 {concept_name} 또는 {column_name}가 특정 {value}와 연결된 경우를 찾습니다.
    # 예: "화면ID가 {scrId}인", "구분코드가 '01'인"
    for p in index["pool"]:
        target_concept = norm(p.get("concept"))
        target_col = norm(p.get("column"))
        expr = p["expr"]

        # Support quoted strings or placeholders like {scrId}
        # re.findall returns list of tuples (full_match, content_if_quoted, content_if_placeholder)
        found_values = re.findall(r"(['\"]([^'\"]+)['\"]|\{([^\}]+)\})", intent)
        for full_val, q_val, p_val in found_values:
            # Check if concept or column name appears near this value in the intent
            label = None
            pos = -1
            
            if target_concept and target_concept in intent:
                label = target_concept
                pos = intent.find(target_concept)
            elif target_col and target_col.lower() in intent.lower():
                label = target_col
                pos = intent.lower().find(target_col.lower())
            
            if label and pos != -1:
                val_pos = intent.find(full_val)
                # If they are somewhat close (within 30 characters)
                if abs(pos - val_pos) < 30:
                    # Avoid duplicates for same target
                    if not any(f["target"] == expr for f in filters):
                        clean_val = p_val if p_val else q_val
                        filters.append({
                            "target": expr,
                            "op": "EQ",
                            "value": full_val if "{" in full_val else clean_val,
                            "valueType": "PLACEHOLDER" if "{" in full_val else "STRING",
                            "confidence": 0.8,
                            "rationale": f"일반 동등 조건 ({label} 매칭)",
                            "evidence": {
                                "source": os.path.basename(INTENT_TXT or "in_memory"),
                                "matched_text": f"{label} ... {full_val}",
                                "rule": "GENERAL_EQ_NAME_MATCH",
                                "span": [min(pos, val_pos), max(pos + len(label), val_pos + len(full_val))]
                            }
                        })

    return {"filters": filters, "dedupe_policy": dedupe_policy}

def llm_validate(intent: str, plan: Dict[str, Any], pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validation only. MUST NOT add new constraints.
    """
    try:
        from openai import OpenAI
    except Exception:
        return {"llm_used": False, "issues": ["openai_sdk_missing"], "suggestions": []}

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"llm_used": False, "issues": ["OPENAI_API_KEY_missing"], "suggestions": []}

    client = OpenAI(api_key=api_key)
    sys = (
        "You are a Text2SQL validator. "
        "Do NOT generate new constraints. "
        "Only validate rule_plan consistency and evidence presence. "
        "Return STRICT JSON with keys: issues (list), suggestions (list)."
    )
    payload = {"intent": intent, "candidate_columns": pool, "rule_plan": plan}
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content": sys},
                      {"role":"user","content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.1,
        )
        txt = resp.choices[0].message.content.strip()
        obj = json.loads(txt)
        obj["llm_used"] = True
        return obj
    except Exception as e:
        return {"llm_used": False, "issues": [f"llm_error:{type(e).__name__}"], "suggestions": []}

def run(intent_txt: str, join_plan_json: str, out_json: str):
    INTENT_TXT = intent_txt
    JOINPLAN_JSON = join_plan_json
    OUT_JSON = out_json
    intent, enc = read_text_robust(INTENT_TXT)

    jp = read_json(JOINPLAN_JSON)
    selected_plan = (jp.get("join_planning") or {}).get("selected_plan") or {}

    index = build_binding_index(selected_plan)

    rule = extract_rules(intent, index, enc)

    limit_val, limit_ev = parse_limit(intent)
    order_by, order_ev = parse_order_by(intent, index["pool"])

    plan = {
        "filters": rule["filters"],
        "time_window": None,
        "group_by": [],
        "having": [],
        "order_by": order_by,
        "limit": limit_val,
        "dedupe_policy": rule["dedupe_policy"],
        "validation": {"llm_used": False, "issues": [], "suggestions": []},
        "trace": {
            "rule_based": True,
            "intent_source": os.path.basename(INTENT_TXT),
            "detected_encoding": enc,
        },
        "evidence_summary": {
            "limit": limit_ev,
            "order_by": order_ev
        }
    }

    if USE_LLM_VALIDATION:
        plan["validation"] = llm_validate(intent, plan, index["pool"])

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print(f"[OK] Agent3 wrote: {OUT_JSON}")
    print(f"  filters={len(plan['filters'])}, limit={plan.get('limit')}, order_by={len(plan.get('order_by') or [])}, dedupe={bool(plan.get('dedupe_policy'))}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python 8_filter_planner_agent_v3.py <intent_txt> <join_plan_json> <out_json>")
    run(sys.argv[1], sys.argv[2], sys.argv[3])



# -----------------------------------------------------------------------------
# In-memory API (no file artifacts required)
# -----------------------------------------------------------------------------
def run_in_memory(intent_text: str, join_plan_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Pure in-memory entrypoint for Agent3."""
    
    # Extract selected_plan if it's the full Agent2 output
    selected_plan = join_plan_obj.get("join_planning", {}).get("selected_plan", {}) or join_plan_obj
    
    idx = build_binding_index(selected_plan)
    pool = idx.get("pool") or []
    
    limit_n, limit_ev = parse_limit(intent_text)
    order_by, order_ev = parse_order_by(intent_text, pool)
    
    rules = extract_rules(intent_text, idx, "utf-8")
    filters = rules.get("filters") or []
    dedupe_policy = rules.get("dedupe_policy")

    out = {
        "filters": filters,
        "limit": limit_n,
        "order_by": order_by,
        "dedupe_policy": dedupe_policy,
        "evidence": {
            "limit": limit_ev,
            "order_by": order_ev,
        },
    }
    return out

# -----------------------------------------------------------------------------
# Backward-compatible file entrypoint
# -----------------------------------------------------------------------------
def run() -> None:
    if not INTENT_TXT or not JOINPLAN_JSON or not OUT_JSON:
        raise RuntimeError("INTENT_TXT/JOINPLAN_JSON/OUT_JSON must be set")
    intent_text, _ = read_text_robust(INTENT_TXT)
    join_plan_obj = read_json(JOINPLAN_JSON)
    out = run_in_memory(intent_text, join_plan_obj)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
