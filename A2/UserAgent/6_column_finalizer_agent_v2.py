# -*- coding: utf-8 -*-
"""
Agent 1: Column Finalizer Agent (post-LLM-finalize "정리" 단계)

What this agent does (fixed scope)
- Input: Step5 output JSON (5.llm_finalize_using_intent_v2.py의 결과물)
- Output: Agent2(Table&Join Planner)의 입력으로 쓸 "selected_columns" JSON 생성

Design principles
- NO new reasoning about column semantics.
- Use final_selection_llm if present; else fallback to pendingSelection; else top1 candidate.
- Preserve auditability: keep source fields + needsHumanReview flags.

Usage
  python 6_column_finalizer_agent_v1.py \
      --in_json  out/step5_finalized.json \
      --out_json out/agent1_selected_columns.json

Output schema (agent2 input)
{
  "selected_columns": [
    {
      "concept": "<original column_candidate>",
      "physicalName": "<selected physical col>",
      "termUri": "<selected termUri or ''>",
      "confidence": 0.0~1.0,
      "needsHumanReview": true|false,
      "source": "final_selection_llm|pendingSelection|top1_candidate",
      "rationale": "<string>"
    }
  ],
  "meta": { "n_items": 12 }
}
"""
from __future__ import annotations

import os
import argparse
import json
from typing import Any, Dict, List, Optional


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _fallback_from_pending(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ps = item.get("pendingSelection") or item.get("pending_selection") or {}
    phys = _norm(ps.get("physicalName") or ps.get("selectedPhysicalName") or ps.get("selected_physical_name"))
    term = _norm(ps.get("termUri") or ps.get("selectedTermUri") or ps.get("term_uri"))
    if not phys:
        return None
    return {
        "physicalName": phys,
        "termUri": term,
        "confidence": 0.0,
        "needsHumanReview": True,
        "source": "pendingSelection",
        "rationale": "FALLBACK_TO_PENDING_SELECTION"
    }


def _fallback_top1_candidate(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ranked = item.get("ranked_candidates") or item.get("candidates") or []
    if not ranked or not isinstance(ranked, list):
        return None
    top = ranked[0] if isinstance(ranked[0], dict) else None
    if not top:
        return None
    phys = _norm(top.get("physicalName") or top.get("physical_column_name") or top.get("물리명") or top.get("physical"))
    term = _norm(top.get("termUri") or top.get("term_uri") or top.get("용어URI") or "")
    if not phys:
        return None
    return {
        "physicalName": phys,
        "termUri": term,
        "confidence": 0.0,
        "needsHumanReview": True,
        "source": "top1_candidate",
        "rationale": "FALLBACK_TO_TOP1_RANKED_CANDIDATE"
    }


def _from_final_selection_llm(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse step5's LLM final selection.

    Support multiple schema variants:
    - Variant A (old): final_selection_llm.{selectedPhysicalName,selectedTermUri,confidence,flags{needsHumanReview},rationale}
    - Variant B (current in 5_final_selection_llm.json):
        final_selection_llm.llm.{selectedPhysicalName,selectedTermUri,confidence,flags{needsHumanReview},rationale}
        final_selection_llm.selected.{physicalName,termUri,score,specificity,...}
    """
    f = item.get("final_selection_llm") or {}

    # ---- physical / term ----
    phys = _norm(
        f.get("selectedPhysicalName")
        or f.get("physicalName")
        or f.get("selected_physical_name")
        or _safe_get(f, ["llm", "selectedPhysicalName"], default=None)
        or _safe_get(f, ["selected", "physicalName"], default=None)
        or _safe_get(f, ["selected", "candidate_physical"], default=None)
    )
    term = _norm(
        f.get("selectedTermUri")
        or f.get("termUri")
        or f.get("term_uri")
        or _safe_get(f, ["llm", "selectedTermUri"], default=None)
        or _safe_get(f, ["selected", "termUri"], default=None)
        or _safe_get(f, ["selected", "term_uri"], default=None)
    )

    # ---- confidence ----
    conf = (
        f.get("confidence")
        if f.get("confidence") is not None
        else _safe_get(f, ["llm", "confidence"], default=None)
    )
    try:
        conf_val = float(conf) if conf is not None else 0.0
    except Exception:
        conf_val = 0.0

    # ---- needsHumanReview ----
    needs = _safe_get(f, ["flags", "needsHumanReview"], default=None)
    if needs is None:
        needs = _safe_get(f, ["llm", "flags", "needsHumanReview"], default=None)
    if needs is None:
        needs = True  # conservative default

    # ---- rationale ----
    rationale = _norm(
        f.get("rationale")
        or _safe_get(f, ["llm", "rationale"], default=None)
        or f.get("decisionRationale")
        or f.get("reason")
    )

    if not phys:
        return None

    return {
        "physicalName": phys,
        "termUri": term,
        "confidence": conf_val,
        "needsHumanReview": bool(needs),
        "source": "final_selection_llm",
        "rationale": rationale or "FINALIZED_BY_LLM"
    }


def finalize_columns(step5_json: Dict[str, Any]) -> Dict[str, Any]:
    results = step5_json.get("results") or []
    selected: List[Dict[str, Any]] = []

    for item in results:
        concept = _norm(item.get("column_candidate") or item.get("columnCandidate") or item.get("concept"))

        packed = (
            _from_final_selection_llm(item)
            or _fallback_from_pending(item)
            or _fallback_top1_candidate(item)
        )

        if not packed:
            selected.append({
                "concept": concept,
                "physicalName": "",
                "termUri": "",
                "confidence": 0.0,
                "needsHumanReview": True,
                "source": "none",
                "rationale": "NO_SELECTION_AVAILABLE"
            })
            continue

        selected.append({"concept": concept, **packed})

    return {"selected_columns": selected, "meta": {"n_items": len(selected)}}


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    IN_JSON = os.path.join(BASE_DIR, "temp_artifacts", "5_final_selection_llm.json")
    OUT_JSON = os.path.join(BASE_DIR, "temp_artifacts", "6_column_finalizer_agent.json")

    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = finalize_columns(data)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {OUT_JSON} (n_items={out['meta']['n_items']})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python 6_column_finalizer_agent_v2.py <in_json> <out_json>")
    main(sys.argv[1], sys.argv[2])
