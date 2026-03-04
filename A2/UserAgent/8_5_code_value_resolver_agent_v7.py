# -*- coding: utf-8 -*-
"""
Agent 8.5 (v7): JSON codebook 기반 CodeValueResolverAgent
- column_to_codeid 제거 (사용자 요구: 100% column name == code_id)
- 근거 기반 유지: codebook에 명시된 라벨->코드 매핑이 있을 때만 치환
- 표기 차이 대응: 정규화 후 exact match 허용
- Fuzzy(유사) 매칭은 자동 치환 금지: 후보만 제공(needsHumanReview)

입력(고정 경로)
- <BASE_DIR>/temp_artifacts/8_constraint_plan.json
- <BASE_DIR>/ttl/codebook_from_code_collect.json   (필수)

출력(고정 경로)
- <BASE_DIR>/temp_artifacts/8_constraint_plan_resolved.json
"""

from __future__ import annotations
import os, json, re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_JSON = ""
OUT_JSON = ""

# ✅ 고정: codebook은 ttl 폴더에 둔다
CODEBOOK_JSON_PATH = os.path.join(BASE_DIR, "ttl", "codebook_from_code_collect.json")

# no-arg knobs
FUZZY_TOPN = 5
FUZZY_MIN_SCORE = 0.85  # 후보 품질 필터(자동치환 X)

def norm(x: Any) -> str:
    return "" if x is None else str(x).strip()

def is_code_column(expr: str) -> bool:
    col = expr.split(".")[-1].upper()
    return col.endswith("_TCD") or col.endswith("_CD") or col.endswith("CD") or col.endswith("TCD")

def ensure_mapping_block(f: Dict[str, Any]) -> Dict[str, Any]:
    if "mapping" not in f or not isinstance(f["mapping"], dict):
        f["mapping"] = {}
    return f

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_korean_label(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

# ---- normalisation / candidates (표기차이 대응) ----
def norm_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)  # whitespace fold
    s = s.replace(" (", "(").replace("( ", "(").replace(" )", ")").replace(") ", ")")
    s = s.replace("ㆍ", "·")
    return s

def build_normalised_label_index(label_to_code: Dict[str, Any]) -> Dict[str, Any]:
    idx: Dict[str, Any] = {}
    for k, v in (label_to_code or {}).items():
        nk = norm_label(k)
        if nk and nk not in idx:
            idx[nk] = v
    return idx

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def suggest_candidates(label_to_code: Dict[str, Any], input_label: str, topn: int = FUZZY_TOPN) -> List[Dict[str, Any]]:
    in_n = norm_label(input_label)
    cands: List[Dict[str, Any]] = []
    for k, v in (label_to_code or {}).items():
        score = sim(in_n, norm_label(k))
        cands.append({"label": k, "code": v, "score": round(score, 4)})
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:topn]

def map_label_to_code_strict_with_normalisation(
    codebook: Dict[str, Any], code_id: str, label: str
) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
    if code_id not in codebook:
        return None, "UNRESOLVED_NO_CODE_ID", []

    m = codebook[code_id].get("label_to_code") or {}

    # raw exact
    if label in m:
        return m[label], "MAPPED_EXACT", []

    # normalised exact
    idx = build_normalised_label_index(m)
    nl = norm_label(label)
    if nl in idx:
        return idx[nl], "MAPPED_EXACT_NORMALISED", []

    # fuzzy candidates (NO auto map)
    cands = suggest_candidates(m, label, topn=FUZZY_TOPN)
    good = [c for c in cands if c.get("score", 0.0) >= FUZZY_MIN_SCORE]
    return None, ("UNRESOLVED_WITH_CANDIDATES" if good else "UNRESOLVED_NO_CANDIDATES"), (good if good else cands)

def resolve_filter(
    f: Dict[str, Any],
    codebook: Dict[str, Any],
    codebook_path: str,
) -> Dict[str, Any]:
    f = ensure_mapping_block(f)
    target = norm(f.get("target"))
    value = f.get("value")

    if not target or not is_code_column(target):
        f["mapping"].update({"status": "SKIPPED_NOT_CODE_COLUMN"})
        return f

    # ✅ 사용자 요구: column name == code_id (엄격)
    code_id = target.split(".")[-1]

    # evidence: code_id 결정 근거(고정 규칙)
    f["mapping"].setdefault("evidence", {})
    f["mapping"]["evidence"]["code_id"] = code_id
    f["mapping"]["evidence"]["code_id_source"] = "STRICT_COLUMN_NAME"

    # string label
    if isinstance(value, str) and is_korean_label(value):
        code, status, cands = map_label_to_code_strict_with_normalisation(codebook, code_id, value)
        if code is not None:
            f["display_value"] = value
            f["value"] = code
            f["valueType"] = "CODE"
            f["mapping"].update({
                "status": status,
                "evidence": {
                    "source": os.path.basename(codebook_path),
                    "code_id": code_id,
                    "label": value,
                    "label_normalised": norm_label(value),
                    "code": code
                }
            })
        else:
            f["mapping"].update({
                "status": status,
                "issues": ["no_exact_label_to_code_mapping_in_codebook_for_code_id"],
                "evidence": {
                    "source": os.path.basename(codebook_path),
                    "code_id": code_id,
                    "label": value,
                    "label_normalised": norm_label(value),
                },
                "candidates": cands
            })
            f["needsHumanReview"] = True
        return f

    # list label
    if isinstance(value, list) and all(isinstance(v, str) for v in value) and any(is_korean_label(v) for v in value):
        codes: List[str] = []
        mapped_pairs: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []

        for v in value:
            if not is_korean_label(v):
                unresolved.append({"label": v, "reason": "non_korean_label"})
                continue
            code, status, cands = map_label_to_code_strict_with_normalisation(codebook, code_id, v)
            if code is None:
                unresolved.append({"label": v, "status": status, "candidates": cands})
            else:
                codes.append(code)
                mapped_pairs.append({"label": v, "label_normalised": norm_label(v), "code": code, "status": status})

        if mapped_pairs and not unresolved:
            f["display_value"] = value
            f["value"] = codes
            f["valueType"] = "LIST_CODE"
            f["mapping"].update({
                "status": "MAPPED",
                "evidence": {"source": os.path.basename(codebook_path), "code_id": code_id, "mapped_pairs": mapped_pairs}
            })
        else:
            f["mapping"].update({
                "status": "PARTIAL" if mapped_pairs else "UNRESOLVED",
                "issues": ["labels_unmapped_or_non_label_values_present"],
                "evidence": {
                    "source": os.path.basename(codebook_path),
                    "code_id": code_id,
                    "mapped_pairs": mapped_pairs,
                    "unresolved": unresolved
                }
            })
            f["needsHumanReview"] = True
        return f

    f["mapping"].update({"status": "SKIPPED_NON_LABEL_VALUE"})
    return f

def run(in_json: str, out_json: str, codebook_json: str | None = None):
    IN_JSON = in_json
    OUT_JSON = out_json
    global CODEBOOK_JSON
    if codebook_json is not None:
        CODEBOOK_JSON = codebook_json
    with open(IN_JSON, "r", encoding="utf-8") as f:
        plan = json.load(f)

    if not os.path.exists(CODEBOOK_JSON_PATH):
        raise FileNotFoundError(f"codebook not found. expected: {CODEBOOK_JSON_PATH}")

    codebook = load_json(CODEBOOK_JSON_PATH)

    filters = plan.get("filters") or []
    resolved_filters: List[Dict[str, Any]] = []
    needs = 0

    for flt in filters:
        if not isinstance(flt, dict):
            continue
        r = resolve_filter(dict(flt), codebook, CODEBOOK_JSON_PATH)
        resolved_filters.append(r)
        if r.get("needsHumanReview"):
            needs += 1

    out = dict(plan)
    out["filters"] = resolved_filters
    out["resolution"] = {
        "codebook": os.path.basename(CODEBOOK_JSON_PATH),
        "codebook_code_ids": len(codebook),
        "n_filters": len(filters),
        "n_needs_human_review": needs,
        "notes": "Auto-mapping only for exact match (raw or normalised). Fuzzy matching produces candidates only (no substitution). code_id is ALWAYS column name."
    }
    out["_trace_resolver"] = {"agent": "8.5", "version": "v7"}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {OUT_JSON}")
    print(f"  codebook={CODEBOOK_JSON_PATH}, code_ids={len(codebook)}, needsHumanReview={needs}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python 8_5_code_value_resolver_agent_v7.py <in_json> <out_json> [codebook_json]")
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)



# -----------------------------------------------------------------------------
# In-memory API (no file artifacts required)
# -----------------------------------------------------------------------------
def run_in_memory(constraints_obj: Dict[str, Any], codebook: Dict[str, Any]) -> Dict[str, Any]:
    """Pure in-memory entrypoint for Agent 8.5."""
    obj = json.loads(json.dumps(constraints_obj))  # deep copy
    filters = obj.get("filters") or []
    for f in filters:
        target = norm(f.get("target") or "")
        value = f.get("value")
        if not isinstance(value, str):
            continue
        # Only map Korean labels for code-like columns
        if not is_code_column(target):
            continue
        # Extract code_id from target colname
        col = target.split(".")[-1].upper()
        code_id = col  # per policy in this agent
        mapped, status, cands = map_label_to_code_strict_with_normalisation(codebook, code_id, value)
        ensure_mapping_block(f)
        f["mapping"]["code_id"] = code_id
        f["mapping"]["input_label"] = value
        f["mapping"]["status"] = status
        if mapped is not None:
            f["mapping"]["mapped_code"] = mapped
            f["value"] = mapped
        else:
            f["mapping"]["candidates"] = cands
    return obj

# -----------------------------------------------------------------------------
# Backward-compatible file entrypoint
# -----------------------------------------------------------------------------
def run() -> None:
    if not IN_JSON or not OUT_JSON:
        raise RuntimeError("IN_JSON/OUT_JSON must be set")
    constraints_obj = load_json(IN_JSON)
    codebook = load_json(CODEBOOK_JSON_PATH)
    out = run_in_memory(constraints_obj, codebook)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
