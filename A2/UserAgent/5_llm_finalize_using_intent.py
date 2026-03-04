# -*- coding: utf-8 -*-
"""
LLM Finalizer (intent-aware, checkpoint-prior, evidence-gated overrides)

Purpose (for your paper + production)
- Pending stage already produced a strong semantic checkpoint:
    - ranked_candidates + rule_score + TTL evidence
    - pending selection (usually GENERIC) that should be *stable* across TTL iterations
- LLM is useful for *disambiguation* with user intent, BUT:
    - user intent must NOT redefine column semantics by default
    - especially: GENERIC -> SPECIALIZED upgrades must be evidence-based and rare

Key changes vs v1
1) Use pending selection as a strong prior (default decision).
2) Allow overriding pending only if LLM provides a verifiable evidence quote from TTL fields.
3) If evidence quote is missing / unverifiable -> automatically fall back to pending selection.
4) No keyword lists, no hardcoded concept exceptions.
5) Hardcoded I/O (no argparse).

Inputs (hardcoded)
- interim_result/4_selection_results.json
- interim_result/UserCanonicalRewrite.txt

Output (hardcoded)
- interim_result/5_final_selection_llm.json
"""

import json
import os
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Hardcoded paths
# -----------------------------------------------------------------------------


OPENAI_MODEL = "gpt-4.1-mini"   # adjust to your environment
TOP_K = 8                       # keep prompt small

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def get_phys(c: Dict[str, Any]) -> str:
    return norm_text(c.get("candidate_physical") or c.get("physicalName") or "")

def get_name_ko(c: Dict[str, Any]) -> str:
    return norm_text(c.get("candidate_term_name_ko") or c.get("termNameKo") or "")

def get_specificity(c: Dict[str, Any]) -> str:
    return norm_text(c.get("specificity") or c.get("Specificity") or "").upper() or "UNKNOWN"

def get_contains_generic_tokens(c: Dict[str, Any]) -> List[str]:
    v = c.get("contains_generic_tokens")
    if v is None:
        v = c.get("containsGenericTokens")
    if v is None:
        return []
    if isinstance(v, list):
        return [norm_text(t).upper() for t in v if norm_text(t)]
    return [norm_text(v).upper()] if norm_text(v) else []

def compress_ttl_evidence(ev: Any) -> Dict[str, Any]:
    """Keep only decision-relevant evidence to minimise prompt size."""
    if not isinstance(ev, dict):
        return {}
    keep = [
        "nameKo",
        "physicalName",
        "notation",
        "definition",
        "detailDescription",
        "scopeInclude",
        "scopeExclude",
        "counterExample",
        "searchTextKo",
        "searchText",
        "specificity",
        "containsGenericTokens",
        "tokens",
    ]
    out: Dict[str, Any] = {}
    for k in keep:
        if k in ev and ev[k]:
            out[k] = ev[k]
    return out

def read_text_auto(path: str) -> str:
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    with open(path, "r", errors="ignore") as f:
        return f.read().strip()

def get_pending_selected(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fs = item.get("final_selection") or {}
    sel = fs.get("selected")
    if not isinstance(sel, dict):
        return None
    return {
        "physicalName": norm_text(sel.get("physicalName")),
        "termNameKo": norm_text(sel.get("termNameKo")),
        "termUri": norm_text(sel.get("termUri")),
        "score": safe_float(sel.get("score", 0.0)),
    }

def build_llm_input(user_intent: str, item: Dict[str, Any]) -> Dict[str, Any]:
    concept = norm_text(item.get("column_candidate"))
    ranked = item.get("ranked_candidates") or []
    ranked = sorted(ranked, key=lambda x: safe_float(x.get("rule_score", -999)), reverse=True)[:TOP_K]

    candidates = []
    for c in ranked:
        ev = compress_ttl_evidence(c.get("ttl_evidence") or {})
        candidates.append({
            "physicalName": get_phys(c),
            "termNameKo": get_name_ko(c),
            "termUri": norm_text(c.get("term_uri") or c.get("termUri") or ""),
            "score": safe_float(c.get("rule_score", 0.0)),
            "specificity": get_specificity(c),
            "containsGenericTokens": get_contains_generic_tokens(c),
            "ttlEvidence": ev
        })

    return {
        "userIntent": user_intent,
        "columnCandidate": concept,
        "pendingSelection": get_pending_selected(item),   # strong prior
        "candidates": candidates
    }

# -----------------------------------------------------------------------------
# LLM prompts (v2)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the final decision maker for mapping a column candidate to exactly one ontology term.

You must choose ONE candidate from the provided list.
You must treat 'pendingSelection' as the DEFAULT unless you can justify an override with TTL evidence.

You may use:
- userIntent (canonical rewrite / expanded intent)
- columnCandidate
- pendingSelection (prior)
- candidates[] evidence and scores (TTL fields in ttlEvidence)

Do NOT use external knowledge. Do NOT invent new terms.
Do NOT apply keyword lists. Do NOT create hardcoded exceptions.

HARD CONSTRAINTS (must follow):
A) Code-type precedence:
   If columnCandidate is code-like (e.g., the surface form indicates a code field such as ending with "코드"),
   you must prefer candidates whose TTL evidence explicitly indicates a code/identifier type
   (e.g., infoType contains "코드", or notation/searchTextKo clearly states it is a code).
   Do not choose name/description terms unless TTL evidence proves it is not a code field.

B) No specialization without explicit evidence:
   If any GENERIC candidate exists, you MUST NOT override pendingSelection to a SPECIALIZED candidate
   unless the SPECIALIZED candidate is proven equivalent OR the user explicitly states the specialization
   and the chosen SPECIALIZED candidate’s TTL evidence explicitly supports that constraint.

Decision hierarchy:
1) Column semantics > TTL evidence > pendingSelection > userIntent.
   - userIntent is supportive context; it must NOT redefine the column semantics by default.
2) Prefer GENERIC when a GENERIC candidate exists, unless a SPECIALIZED candidate is proven equivalent.

Override policy (critical):
- If you override pendingSelection, you MUST provide a short 'evidenceQuote' copied verbatim from the chosen candidate's TTL evidence
  (definition/scopeInclude/scopeExclude/detailDescription/searchTextKo).
- evidenceQuote should preferentially come from definition/scopeInclude/scopeExclude (searchTextKo is fallback).
- The evidenceQuote must demonstrate why the override is correct (equivalence/constraint).
- If you cannot provide a verifiable quote, do NOT override pendingSelection.

Output JSON only.
"""

USER_TEMPLATE = """Input JSON:
{payload}

Return JSON only with schema:
{{
  "selectedPhysicalName": string,
  "selectedTermUri": string,
  "confidence": number,
  "rationale": string,
  "overridePending": boolean,
  "evidenceSource": "definition"|"scopeInclude"|"scopeExclude"|"detailDescription"|"searchTextKo"|null,
  "evidenceQuote": string|null,
  "flags": {{
    "preferredGenericOverSpecialized": boolean,
    "specializedChosen": boolean,
    "needsHumanReview": boolean
  }}
}}

Hard constraints:
- selectedPhysicalName must be one of candidates[].physicalName.
- If overridePending=true, evidenceSource and evidenceQuote MUST be non-null.
- If confidence < 0.65 => needsHumanReview=true.
"""

def call_llm(payload: Dict[str, Any]) -> Dict[str, Any]:
    from openai import OpenAI
    
    api_key = ""

    
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False, indent=2))}
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def _evidence_text(ev: Dict[str, Any], source: str) -> str:
    v = ev.get(source)
    if v is None:
        return ""
    if isinstance(v, list):
        return "\n".join([norm_text(x) for x in v if norm_text(x)])
    return norm_text(v)

def _force_fallback_pending(reason: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Force selection to pendingSelection if available, else top1 candidate."""
    cand_map = {c["physicalName"]: c for c in payload["candidates"]}
    pending = payload.get("pendingSelection") or None

    chosen = None
    if pending and pending.get("physicalName") and pending["physicalName"] in cand_map:
        chosen = cand_map[pending["physicalName"]]
    elif payload["candidates"]:
        chosen = payload["candidates"][0]

    if not chosen:
        return {
            "selectedPhysicalName": "",
            "selectedTermUri": "",
            "confidence": 0.0,
            "rationale": "Fallback: no candidates.",
            "overridePending": False,
            "evidenceSource": None,
            "evidenceQuote": None,
            "flags": {
                "preferredGenericOverSpecialized": False,
                "specializedChosen": False,
                "needsHumanReview": True
            }
        }

    spec = chosen.get("specificity")
    return {
        "selectedPhysicalName": chosen["physicalName"],
        "selectedTermUri": chosen.get("termUri") or "",
        "confidence": 0.0,
        "rationale": f"FORCED_FALLBACK_TO_PENDING: {reason}",
        "overridePending": False,
        "evidenceSource": None,
        "evidenceQuote": None,
        "flags": {
            "preferredGenericOverSpecialized": spec == "GENERIC",
            "specializedChosen": spec == "SPECIALIZED",
            "needsHumanReview": True
        }
    }

def validate_and_apply_gates(llm: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate output and enforce evidence-gated override."""
    cand_map = {c["physicalName"]: c for c in payload["candidates"]}
    pending = payload.get("pendingSelection") or None

    sel = llm.get("selectedPhysicalName")
    if sel not in cand_map:
        return _force_fallback_pending("Invalid selection not in candidates.", payload)

    if not llm.get("selectedTermUri"):
        llm["selectedTermUri"] = cand_map[sel].get("termUri") or ""

    llm["flags"] = llm.get("flags") or {}
    chosen_spec = cand_map[sel].get("specificity")
    llm["flags"].setdefault("specializedChosen", chosen_spec == "SPECIALIZED")
    llm["flags"].setdefault("preferredGenericOverSpecialized", chosen_spec == "GENERIC")
    llm["flags"]["needsHumanReview"] = bool(llm["flags"].get("needsHumanReview") or safe_float(llm.get("confidence", 0.0)) < 0.65)

    if pending and norm_text(pending.get("physicalName")) and sel != pending["physicalName"]:
        llm["overridePending"] = True

        src = llm.get("evidenceSource")
        quote = llm.get("evidenceQuote")

        if not src or not quote:
            return _force_fallback_pending("Override requested but evidenceSource/evidenceQuote missing.", payload)

        ev = cand_map[sel].get("ttlEvidence") or {}
        hay = _evidence_text(ev, src)

        if quote not in hay:
            return _force_fallback_pending("Override requested but evidenceQuote not found in TTL evidence.", payload)

        # Additional safety: block GENERIC->SPECIALIZED upgrades unless confidence is high
        pending_phys = pending["physicalName"]
        pending_cand = cand_map.get(pending_phys)  # may be absent if not in top_k
        pending_spec = pending_cand.get("specificity") if pending_cand else None

        if pending_spec == "GENERIC" and chosen_spec == "SPECIALIZED":
            if safe_float(llm.get("confidence", 0.0)) < 0.80:
                return _force_fallback_pending("Blocked GENERIC->SPECIALIZED upgrade: confidence < 0.80.", payload)

    else:
        llm["overridePending"] = False
        llm.setdefault("evidenceSource", None)
        llm.setdefault("evidenceQuote", None)

    return llm

def pack_selected(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "physicalName": c.get("physicalName"),
        "termNameKo": c.get("termNameKo"),
        "termUri": c.get("termUri"),
        "score": c.get("score"),
        "specificity": c.get("specificity"),
        "contains_generic_tokens": c.get("containsGenericTokens"),
    }

def fallback_top1(item: Dict[str, Any]) -> Dict[str, Any]:
    ranked = item.get("ranked_candidates") or []
    ranked = sorted(ranked, key=lambda x: safe_float(x.get("rule_score", -999)), reverse=True)
    if not ranked:
        return {
            "selected": None,
            "llm": {
                "selectedPhysicalName": "",
                "selectedTermUri": "",
                "confidence": 0.0,
                "rationale": "Fallback: no candidates",
                "overridePending": False,
                "evidenceSource": None,
                "evidenceQuote": None,
                "flags": {
                    "preferredGenericOverSpecialized": False,
                    "specializedChosen": False,
                    "needsHumanReview": True
                }
            }
        }
    top1 = ranked[0]
    selected = {
        "physicalName": get_phys(top1),
        "termNameKo": get_name_ko(top1),
        "termUri": norm_text(top1.get("term_uri") or top1.get("termUri") or ""),
        "score": safe_float(top1.get("rule_score", 0.0)),
        "specificity": get_specificity(top1),
        "contains_generic_tokens": get_contains_generic_tokens(top1),
    }
    return {
        "selected": selected,
        "llm": {
            "selectedPhysicalName": selected["physicalName"],
            "selectedTermUri": selected["termUri"],
            "confidence": 0.0,
            "rationale": "Fallback: OPENAI_API_KEY missing. Used top1.",
            "overridePending": False,
            "evidenceSource": None,
            "evidenceQuote": None,
            "flags": {
                "preferredGenericOverSpecialized": (selected["specificity"] == "GENERIC"),
                "specializedChosen": (selected["specificity"] == "SPECIALIZED"),
                "needsHumanReview": True
            }
        }
    }

def main_llm_finalize_using_intent(user_intent, in_data):
    
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    #IN_JSON = os.path.join(BASE_DIR, "interim_result", "4_selection_results.json")
    #INTENT_TXT = os.path.join(BASE_DIR, "interim_result", "UserCanonicalRewrite.txt")


    #OUT_JSON = os.path.join(BASE_DIR, "interim_result", "5_final_selection_llm.json")



    #user_intent = read_text_auto(INTENT_TXT) if os.path.exists(INTENT_TXT) else ""

    #with open(IN_JSON, "r", encoding="utf-8") as f:
    #    data = json.load(f)
    data = in_data
    results = data.get("results", [])
    out_results = []

    can_call_llm = True

    for item in results:
        payload = build_llm_input(user_intent, item)

        if not can_call_llm:
            packed = fallback_top1(item)
            out_results.append({**item, "final_selection_llm": {"decisionType": "FINALIZED_BY_LLM_V2", **packed}})
            continue

        try:
            llm = call_llm(payload)
            llm = validate_and_apply_gates(llm, payload)

            cand_map = {c["physicalName"]: c for c in payload["candidates"]}
            chosen = cand_map.get(llm["selectedPhysicalName"])
            selected = pack_selected(chosen) if chosen else None

            out_results.append({
                **item,
                "final_selection_llm": {
                    "decisionType": "FINALIZED_BY_LLM_V2",
                    "selected": selected,
                    "llm": llm
                }
            })
        except Exception as e:
            forced = _force_fallback_pending(f"LLM error: {type(e).__name__}: {e}", payload)
            cand_map = {c["physicalName"]: c for c in payload["candidates"]}
            chosen = cand_map.get(forced["selectedPhysicalName"])
            selected = pack_selected(chosen) if chosen else None
            out_results.append({
                **item,
                "final_selection_llm": {
                    "decisionType": "FINALIZED_BY_LLM_V2",
                    "selected": selected,
                    "llm": forced
                }
            })

    out = {"results": out_results}

    return out

    #with open(OUT_JSON, "w", encoding="utf-8") as f:
    #    json.dump(out, f, ensure_ascii=False, indent=2)

    #print("Wrote:", OUT_JSON)

#if __name__ == "__main__":
#    main()
