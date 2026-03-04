# -*- coding: utf-8 -*-
"""
Semantic Checkpoints version (v4) based on 4.OntologySearchSelection_v3_trace_slim.py

What changed (high level)
- Uses existing LLM term description fields already stored in TTL (ex:searchTextKo / ex:searchText) as retrieval signals.
- Removes "LLM selects Top1" step. Instead:
  1) Retrieve + score candidates (embedding ∪ ontology).
  2) Build Semantic Checkpoints ONLY for high-impact & high-uncertainty links.
  3) Either:
     - output checkpoints for later UI/HITL (default), or
     - run interactive CLI confirmation, or
     - auto-select with SQL policy (no questions).
- includeScope / excludeScope / counterExamples (often LLM-generated) are NOT used for scoring/selection.
  They are used only as WARNING/TRIGGER signals for checkpoints (never as hard reject).

Inputs
- 3_benchmark.json: {"mappings":[{"column_candidate": "...", "candidates":[...]}]}
- TTL: local turtle file

Outputs
- 4_selection_results_semantic_checkpoints.json:
  - proposals (ranked candidates + trace)
  - semantic_checkpoints (0..N)
  - final_selection (auto or confirmed)
"""

import os
import re
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set

from rdflib import Graph
from rdflib.namespace import RDF, RDFS, SKOS, Namespace
from rdflib.term import URIRef, Literal

def phys_tokens(physical: str) -> Set[str]:
    """
    Tokenize physical name like 'BFPROC_AC_STAT_TCD' or 'AC_STAT_TCD'
    into {'BFPROC','AC','STAT','TCD'} (uppercase).
    """
    p = norm_text(physical).upper()
    if not p:
        return set()
    # split by non-alnum (underscore etc.)
    parts = re.split(r"[^A-Z0-9]+", p)
    return {x for x in parts if x}


# -----------------------------
# Utilities
# -----------------------------
def norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()

def fold_for_search(s: str) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = re.sub(r"[^\w\s가-힣]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s



def fold_for_key(s: str) -> str:
    """Stronger normalization for label-key matching (whitespace/punct removed)."""
    s = norm_text(s).lower()
    s = re.sub(r"[\s\-_]+", "", s)          # remove spaces / hyphens / underscores
    s = re.sub(r"[^\w가-힣]", "", s)          # remove punctuation (keep Hangul)
    return s.strip()


def build_label_hash_index(term_index: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build O(1) label->term candidates index (no full-scan search at runtime)."""
    h: Dict[str, List[Dict[str, Any]]] = {}

    def _add(key: str, t: Dict[str, Any], kind: str) -> None:
        if not key:
            return
        h.setdefault(key, []).append({
            "source": "ontology_anchor",
            "_stage": f"anchor_{kind}",
            "용어명": t.get("term_name_ko", "") or "",
            "물리명": t.get("physical", "") or "",
            "_term_uri": t.get("term_uri", "") or "",
        })

    for t in term_index:
        prof: TermProfile = t.get("profile")
        if not prof:
            continue

        # labels
        for lab in (prof.pref_labels_ko or []):
            _add(fold_for_key(lab), t, "pref")
        for lab in (prof.alt_labels_ko or []):
            _add(fold_for_key(lab), t, "alt")

        # code-ish identifiers
        for lab in (prof.hidden_labels or []):
            _add(fold_for_key(lab), t, "hidden")
        for lab in (prof.notation or []):
            _add(fold_for_key(lab), t, "notation")

        # physical (best-effort)
        phys = t.get("physical") or ""
        if phys:
            _add(fold_for_key(phys), t, "physical")

    return h


def ontology_anchor_candidates(column_candidate: str, label_index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Inject ontology candidates via exact label/notation hits (cheap + deterministic)."""
    q = norm_text(column_candidate)
    if not q:
        return []
    key = fold_for_key(q)
    return list(label_index.get(key, []))
def ko_en_num_tokens(s: str) -> List[str]:
    s = fold_for_search(s)
    if not s:
        return []
    toks = re.findall(r"[가-힣]+|[a-z]+|\d+", s)
    return [t for t in toks if t]

def uniq_preserve(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = norm_text(x)
        if not x or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out

def safe_first(xs: List[str]) -> str:
    return xs[0] if xs else ""

def shorten_text(s: str, max_chars: int) -> str:
    s = norm_text(s)
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)].rstrip() + "…"

# -----------------------------
# TTL Term profile
# -----------------------------
@dataclass
class TermProfile:
    term_uri: str
    pref_labels_ko: List[str]
    alt_labels_ko: List[str]
    definition_ko: List[str]
    detail_desc_ko: List[str]
    scope_include_ko: List[str]
    scope_exclude_ko: List[str]
    counter_examples_ko: List[str]
    hidden_labels: List[str]
    notation: List[str]
    tokens: List[str]
    search_text_ko: Optional[str]   # typically "LLM term description (ko)" in your TTL
    search_text: Optional[str]      # typically "LLM term description (en)" in your TTL
    specificity: Optional[str]      # EX:specificity (GENERIC|SPECIALIZED)
    contains_generic_tokens: List[str]  # EX:containsGenericToken (e.g., CIF, ACNO)

def lit_texts(g: Graph, s: URIRef, p: URIRef, lang: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            if lang is None or o.language == lang:
                v = norm_text(o)
                if v:
                    out.append(v)
    return uniq_preserve(out)

def load_graph(ttl_path: str) -> Graph:
    if not os.path.exists(ttl_path):
        raise FileNotFoundError(f"TTL not found: {ttl_path}")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g

def find_term_uri_by_physical(g: Graph, physical: str) -> Optional[URIRef]:
    physical = norm_text(physical)
    if not physical:
        return None
    for s in g.subjects(SKOS.notation, Literal(physical)):
        return s
    for s in g.subjects(SKOS.hiddenLabel, Literal(physical)):
        return s
    return None

def collect_tokens_from_hasToken(g: Graph, term_uri: URIRef, EX: Namespace) -> List[str]:
    toks: List[str] = []
    for tok_uri in g.objects(term_uri, EX.hasToken):
        if isinstance(tok_uri, URIRef):
            toks += lit_texts(g, tok_uri, RDFS.label, lang=None)
            toks += lit_texts(g, tok_uri, SKOS.prefLabel, lang=None)
            toks += lit_texts(g, tok_uri, SKOS.notation, lang=None)
            toks += lit_texts(g, tok_uri, SKOS.hiddenLabel, lang=None)
    return uniq_preserve(toks)

def load_term_profile(g: Graph, term_uri: URIRef, EX: Namespace) -> TermProfile:
    pref_ko = lit_texts(g, term_uri, SKOS.prefLabel, lang="ko") + lit_texts(g, term_uri, RDFS.label, lang="ko")
    pref_ko = uniq_preserve(pref_ko)

    alt_ko = uniq_preserve(lit_texts(g, term_uri, SKOS.altLabel, lang="ko"))
    def_ko = uniq_preserve(lit_texts(g, term_uri, SKOS.definition, lang="ko"))

    detail_ko = uniq_preserve(lit_texts(g, term_uri, EX.detailDescription, lang="ko"))

    # include/exclude/counterexamples exist but are treated as WARN signals only (not decision factors)
    inc_ko = uniq_preserve(lit_texts(g, term_uri, EX.includeScope, lang="ko"))
    exc_ko = uniq_preserve(lit_texts(g, term_uri, EX.excludeScope, lang="ko"))

    scope_notes = lit_texts(g, term_uri, SKOS.scopeNote, lang="ko")
    for sn in scope_notes:
        up = sn.upper()
        if up.startswith("INCLUDE:"):
            inc_ko.append(sn.split(":", 1)[1].strip())
        elif up.startswith("EXCLUDE:"):
            exc_ko.append(sn.split(":", 1)[1].strip())
        else:
            detail_ko.append(sn)

    inc_ko = uniq_preserve(inc_ko)
    exc_ko = uniq_preserve(exc_ko)
    detail_ko = uniq_preserve(detail_ko)

    ce_ko = uniq_preserve(lit_texts(g, term_uri, EX.counterExample, lang="ko") + lit_texts(g, term_uri, SKOS.example, lang="ko"))

    hidden = uniq_preserve(lit_texts(g, term_uri, SKOS.hiddenLabel, lang=None))
    notation = uniq_preserve(lit_texts(g, term_uri, SKOS.notation, lang=None))

    tokens = uniq_preserve(collect_tokens_from_hasToken(g, term_uri, EX) + hidden + notation)

    st_ko = safe_first(lit_texts(g, term_uri, EX.searchTextKo, lang=None))
    st_en = safe_first(lit_texts(g, term_uri, EX.searchText, lang=None))

    spec = safe_first(lit_texts(g, term_uri, EX.specificity, lang=None))
    cgt = uniq_preserve(lit_texts(g, term_uri, EX.containsGenericToken, lang=None))

    return TermProfile(
        term_uri=str(term_uri),
        pref_labels_ko=pref_ko,
        alt_labels_ko=alt_ko,
        definition_ko=def_ko,
        detail_desc_ko=detail_ko,
        scope_include_ko=inc_ko,
        scope_exclude_ko=exc_ko,
        counter_examples_ko=ce_ko,
        hidden_labels=hidden,
        notation=notation,
        tokens=tokens,
        search_text_ko=st_ko if st_ko else None,
        search_text=st_en if st_en else None,
        specificity=spec if spec else None,
        contains_generic_tokens=cgt,
    )

# -----------------------------
# Ontology index & retrieval
# -----------------------------
def build_term_index(g: Graph, EX: Namespace) -> List[Dict[str, Any]]:
    """
    Build a searchable index over TTL terms.

    IMPORTANT (policy):
    - Retrieval signals may include: labels, definitions, detailDescription, tokens, and existing LLM term descriptions (searchTextKo/searchText).
    - LLM-generated include/exclude/counterexamples are NOT used as retrieval scoring signals here.
      (They will be kept in profile for warnings/checkpoints only.)
    """
    out: List[Dict[str, Any]] = []
    term_uris = set(g.subjects(RDF.type, EX.Term))
    if not term_uris:
        term_uris = set(g.subjects(RDF.type, SKOS.Concept))

    for term_uri in term_uris:
        if not isinstance(term_uri, URIRef):
            continue
        prof = load_term_profile(g, term_uri, EX)

        # physical guess
        physical = ""
        for n in prof.notation + prof.hidden_labels:
            if "_" in n or (n.isupper() and len(n) >= 2):
                physical = n
                break

        blob_parts: List[str] = []
        blob_parts += prof.pref_labels_ko
        blob_parts += prof.alt_labels_ko
        blob_parts += prof.definition_ko
        blob_parts += prof.detail_desc_ko
        blob_parts += prof.hidden_labels
        blob_parts += prof.notation
        blob_parts += prof.tokens

        # existing LLM term descriptions in TTL (your "safe" LLM signal)
        if prof.search_text_ko:
            blob_parts.append(prof.search_text_ko)
        if prof.search_text:
            blob_parts.append(prof.search_text)

        blob = fold_for_search(" | ".join([x for x in blob_parts if x]))

        out.append({
            "term_uri": prof.term_uri,
            "term_name_ko": safe_first(prof.pref_labels_ko) or safe_first(prof.alt_labels_ko),
            "physical": physical,
            "blob": blob,
            "profile": prof,
        })
    return out

def _retrieval_score(query: str, t: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Evidence-based retrieval score (no TopK).
    - Uses only positive signals (substring/token overlap).
    - Does NOT penalize include/exclude/counterexamples (policy: those are unreliable).
    """
    q = fold_for_search(query)
    q_toks = set(ko_en_num_tokens(q))
    blob = t["blob"]

    evidence: Dict[str, Any] = {
        "substr_hit": False,
        "token_overlap": 0.0,
    }

    score = 0.0
    if q and q in blob:
        evidence["substr_hit"] = True
        score += 1.0

    if q_toks:
        blob_toks = set(ko_en_num_tokens(blob))
        if blob_toks:
            overlap = len(q_toks & blob_toks) / max(1, len(q_toks))
            evidence["token_overlap"] = overlap
            score += overlap

    return score, evidence

def pick_elbow_cutoff(sorted_scores: List[float]) -> int:
    n = len(sorted_scores)
    if n <= 1:
        return n - 1
    if n == 2:
        return 1

    diffs = [sorted_scores[i] - sorted_scores[i + 1] for i in range(n - 1)]
    if len(diffs) <= 1:
        return min(1, n - 1)

    ddiffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
    if not ddiffs:
        return min(1, n - 1)

    elbow = max(range(len(ddiffs)), key=lambda i: ddiffs[i]) + 1
    elbow = max(0, min(elbow, n - 1))
    return max(0, elbow)

def ontology_retrieve_candidates(column_candidate: str, term_index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q = norm_text(column_candidate)
    if not q:
        return []

    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for t in term_index:
        s, ev = _retrieval_score(q, t)
        # remove pure-no-signal items
        if s <= 0.0:
            continue
        scored.append((s, t, ev))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    scores_only = [x[0] for x in scored]
    cutoff = pick_elbow_cutoff(scores_only)
    picked = scored[: cutoff + 1]

    out: List[Dict[str, Any]] = []
    for s, t, ev in picked:
        out.append({
            "source": "ontology",
            "_stage": "ontology",
            "_ontology_score": s,
            "_ontology_evidence": ev,
            "용어명": t.get("term_name_ko", ""),
            "물리명": t.get("physical", ""),
            "_term_uri": t.get("term_uri", ""),
        })
    return out

def merge_candidates(embed_cands: List[Dict[str, Any]], onto_cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    def key(c: Dict[str, Any]) -> str:
        phys = norm_text(c.get("물리명", ""))
        uri = norm_text(c.get("_term_uri", ""))
        return phys if phys else uri

    for c in embed_cands or []:
        c = dict(c)
        c["source"] = "embedding"
        k = key(c)
        if not k or k in seen:
            continue
        out.append(c)
        seen.add(k)

    for c in onto_cands or []:
        c = dict(c)
        k = key(c)
        if not k or k in seen:
            continue
        out.append(c)
        seen.add(k)

    return out

# -----------------------------
# Candidate scoring + trace
# -----------------------------
@dataclass
class CandidateScore:
    column_candidate: str
    source: str
    stage: str
    candidate_term_name_ko: str
    candidate_physical: str
    term_uri: str
    rule_score: float
    rule_signals: Dict[str, Any]
    match_trace: Dict[str, Any]
    ttl_evidence: Dict[str, Any]
    specificity: str  # GENERIC|SPECIALIZED|UNKNOWN
    contains_generic_tokens: List[str]
    value_type: str  # CODE/NAME/ID/NO/DATE/OTHER

def build_evidence_dict(prof: TermProfile) -> Dict[str, Any]:
    return {
        "pref_labels_ko": prof.pref_labels_ko,
        "alt_labels_ko": prof.alt_labels_ko,
        "hidden_labels": prof.hidden_labels,
        "notation": prof.notation,
        "definition_ko": prof.definition_ko,
        "detail_desc_ko": prof.detail_desc_ko,
        # WARN-only fields
        "scope_include_ko": prof.scope_include_ko,
        "scope_exclude_ko": prof.scope_exclude_ko,
        "counter_examples_ko": prof.counter_examples_ko,
        # token/LLM desc
        "tokens": prof.tokens,
        "searchTextKo": prof.search_text_ko,
        "searchText": prof.search_text,
        "specificity": prof.specificity,
        "containsGenericTokens": prof.contains_generic_tokens,
    }

def infer_value_type(physical: str) -> str:
    p = norm_text(physical).upper()
    if not p:
        return "OTHER"
    # common enterprise suffix patterns
    if re.search(r"(_TCD|_CD|_CODE|_COD)$", p):
        return "CODE"
    if re.search(r"(_NM|_NAME)$", p):
        return "NAME"
    if re.search(r"(_ID)$", p):
        return "ID"
    if re.search(r"(_NO|_NUM|_NUMBER)$", p):
        return "NO"
    if re.search(r"(_DT|_DATE|_DTTM|_TM|_TIME)$", p):
        return "DATE"
    return "OTHER"

def sql_policy_prefers_code(column_candidate: str) -> bool:
    """
    Default: SQL coding preference = CODE/ID/NO first,
    unless user explicitly requests '명칭/설명/내용/리포팅' etc.
    For the benchmark 'column_candidate' string alone, we assume default is SQL_CODING_DEFAULT.
    """
    q = norm_text(column_candidate)
    if not q:
        return True
    # crude "user requested explanation" detector
    if re.search(r"(설명|내용|명칭|리포팅|레포트|보고서|이름|name|label|desc)", q, re.IGNORECASE):
        return False
    return True

def rule_signals(column_candidate: str, cand: Dict[str, Any], prof: TermProfile) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """
    Explainable, bounded scoring for ordering.
    POLICY:
    - include/exclude/counterexamples are NOT used for penalizing or rejecting.
      They are only recorded as warning triggers for checkpointing.
    """
    col = norm_text(column_candidate)
    col_fold = fold_for_search(col)
    col_toks = set(ko_en_num_tokens(col))

    cand_name = norm_text(cand.get("용어명", "")) or safe_first(prof.pref_labels_ko) or safe_first(prof.alt_labels_ko)
    cand_phys = norm_text(cand.get("물리명", "")) or safe_first(prof.notation) or safe_first(prof.hidden_labels)
    stage = norm_text(cand.get("_stage", ""))

    label_pool = uniq_preserve(
        [cand_name] +
        list(prof.pref_labels_ko) +
        list(prof.alt_labels_ko) +
        list(prof.hidden_labels) +
        list(prof.notation)
    )
    label_pool_fold = [(x, fold_for_search(norm_text(x))) for x in label_pool if norm_text(x)]

    exact_label = False
    substr_label = False
    exact_matches = []
    substr_matches = []

    if col_fold:
        for raw, f in label_pool_fold:
            if not f:
                continue
            if col_fold == f:
                exact_label = True
                exact_matches.append(raw)
            elif (f in col_fold) or (col_fold in f):
                substr_label = True
                substr_matches.append(raw)

    cand_tok_pool = set()
    for x in label_pool:
        cand_tok_pool |= set(ko_en_num_tokens(norm_text(x)))
    cand_tok_pool |= set(ko_en_num_tokens(" ".join(prof.tokens)))

    inter = sorted(list(col_toks & cand_tok_pool))
    tok_overlap = len(inter) / max(1, len(col_toks))

    def_tok_pool = set(ko_en_num_tokens(" ".join(prof.definition_ko + prof.detail_desc_ko)))
    inter2 = sorted(list(col_toks & def_tok_pool))
    def_overlap = len(inter2) / max(1, len(col_toks))

    # warning triggers (NOT scoring)
    warn = {
        "exclude_scope_hit": False,
        "exclude_scope_trigger": None,
        "counterexample_hit": False,
        "counterexample_trigger": None,
    }
    # "hit" is heuristic: if query term itself appears in exclude/counterexample texts
    q_fold = col_fold
    if q_fold:
        for raw in prof.counter_examples_ko:
            rf = fold_for_search(raw)
            if rf and (q_fold in rf or rf in q_fold):
                warn["counterexample_hit"] = True
                warn["counterexample_trigger"] = raw
                break
        if not warn["counterexample_hit"]:
            for raw in prof.scope_exclude_ko:
                rf = fold_for_search(raw)
                if rf and (q_fold in rf or rf in q_fold):
                    warn["exclude_scope_hit"] = True
                    warn["exclude_scope_trigger"] = raw
                    break

    # base score
    # IMPORTANT: use ONLY TTL-backed semantic evidence (definition + scopeNote).
    # Do NOT use label substring/prefix signals, token overlap against labels, or stage hints for scoring.
    score = 0.0

    # MAIN evidence: scopeNote text (termDetailDescription -> skos:scopeNote)
    scope_text = " ".join(prof.detail_desc_ko or [])
    scope_toks = set(ko_en_num_tokens(scope_text))
    scope_overlap = (len(col_toks & scope_toks) / max(1, len(col_toks))) if scope_toks else 0.0
    score += 2.0 * scope_overlap

    # SECOND evidence: definition overlap (already computed above as def_overlap)
    score += 1.0 * def_overlap

# stage hint (tiny)
    # NOTE: stage is derived from retrieval heuristics (exact/prefix/substring/fuzzy/ontology).
    # We intentionally do NOT use it for scoring because it is not TTL semantic evidence.
    # (kept only for tracing/debug prints if needed)
    st = stage.lower()

    # specificity policy (minimal v1) (minimal v1)
    # Goal: avoid premature "SPECIALIZED" selection when the column candidate is generic/underspecified.
    # Uses TTL ex:specificity only (no family/baseTerm logic).
    spec = (prof.specificity or "").strip().upper()
    spec_adjust = 0.0
    if spec == "SPECIALIZED":
        # If the user explicitly wrote the specialized label, we should not fight it.
        # Otherwise, apply a mild penalty so GENERIC can win when both are plausible.
        if not exact_label:
            spec_adjust = -0.40
    elif spec == "GENERIC":
        if not exact_label:
            spec_adjust = +0.20
    score += spec_adjust
    # sql policy tiny preference (ordering only, not hard decision)
    if sql_policy_prefers_code(column_candidate):
        vt = infer_value_type(cand_phys)
        if vt in ("CODE", "ID", "NO"):
            score += 0.10
        elif vt == "NAME":
            score -= 0.05

    signals = {
        "exact_label": exact_label,
        "substring_label": substr_label,
        "token_overlap": tok_overlap,
        "definition_overlap": def_overlap,
        "stage": stage,
        "candidate_physical": cand_phys,
        "candidate_name": cand_name,
        "warnings": warn,
        "specificity": prof.specificity,
        "specificity_adjust": spec_adjust,
    }

    match_trace = {
        "label_matches": {
            "exact": exact_matches[:10],
            "substring": substr_matches[:10],
        },
        "token_overlap": {
            "overlapped_tokens": inter[:50],
        },
        "definition_overlap": {
            "overlapped_tokens": inter2[:50],
        },
        "warning_triggers": warn,
    }

    return score, signals, match_trace

# -----------------------------
# Semantic checkpointing
# -----------------------------
def is_high_impact_concept(concept: str) -> bool:
    """
    Minimal heuristic: concepts that likely become JOIN keys or WHERE filters.
    """
    c = norm_text(concept)
    if not c:
        return False
    # keys / identifiers / state / code
    if re.search(r"(식별|ID|아이디|번호|NO|코드|CODE|상태|구분|여부|일시|일자|DATE)", c, re.IGNORECASE):
        return True
    return False

def compute_uncertainty(top1: CandidateScore, top2: Optional[CandidateScore]) -> Dict[str, Any]:
    """
    Uncertainty signals for checkpoint decision.
    """
    if top2 is None:
        return {"delta": 999.0, "type_conflict": False}

    delta = float(top1.rule_score - top2.rule_score)
    t1 = top1.value_type
    t2 = top2.value_type

    # type conflict: CODE/ID/NO vs NAME
    type_conflict = ((t1 in ("CODE","ID","NO")) and (t2 == "NAME")) or ((t2 in ("CODE","ID","NO")) and (t1 == "NAME"))
    return {"delta": delta, "type_conflict": type_conflict}

def build_semantic_checkpoints(concept: str, ranked: List[CandidateScore], max_options: int = 4) -> List[Dict[str, Any]]:
    """
    Build 0..N checkpoints for this concept. Default is at most 1 checkpoint per concept.
    We ask ONLY when:
    - high impact AND
    - uncertainty high (small delta or type conflict) OR warnings triggered.
    """
    if not ranked:
        return []

    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) >= 2 else None
    unc = compute_uncertainty(top1, top2)

    warnings = top1.rule_signals.get("warnings", {}) if top1.rule_signals else {}
    warn_hit = bool(warnings.get("exclude_scope_hit") or warnings.get("counterexample_hit"))

    ask = False
    reasons = []

    if is_high_impact_concept(concept):
        if top2 is not None and (unc["delta"] < 0.25):
            ask = True
            reasons.append(f"LOW_MARGIN(delta={unc['delta']:.3f})")
        if unc["type_conflict"]:
            ask = True
            reasons.append("TYPE_CONFLICT(CODE/ID/NO vs NAME)")
        if warn_hit:
            ask = True
            reasons.append("ONTOLOGY_WARN_TRIGGER(include/exclude/counterexample)")

    if not ask:
        return []

    options = []
    for cs in ranked[:max_options]:
        ev = cs.ttl_evidence or {}
        options.append({
            "physicalName": cs.candidate_physical,
            "termNameKo": cs.candidate_term_name_ko,
            "termUri": str(cs.term_uri),
            "valueType": cs.value_type,
            "score": cs.rule_score,
            "why": {
                "label_exact": bool(cs.rule_signals.get("exact_label")),
                "label_substr": bool(cs.rule_signals.get("substring_label")),
                "token_overlap": cs.rule_signals.get("token_overlap"),
                "definition_overlap": cs.rule_signals.get("definition_overlap"),
                "warnings": cs.rule_signals.get("warnings", {}),
                "ttl_glance": {
                    "prefLabel": safe_first(ev.get("pref_labels_ko", [])),
                    "definition": shorten_text(safe_first(ev.get("definition_ko", [])), 180),
                    "searchTextKo": shorten_text(norm_text(ev.get("searchTextKo", "")), 220),
                }
            }
        })

    # Build a single checkpoint
    question = (
        f"'{concept}'의 의미/용법이 모호합니다. 아래 후보 중 무엇을 의도하셨나요?\n"
        f"- SQL 코딩 목적이면 일반적으로 CODE/ID/NO 컬럼이 우선입니다.\n"
        f"- '명칭/설명/리포팅' 목적이면 NAME 컬럼이 우선일 수 있습니다."
    )

    return [{
        "checkpoint_id": f"CP_{fold_for_search(concept).replace(' ','_')}",
        "concept": concept,
        "reasons": reasons,
        "question": question,
        "options": options,
        "recommended": {
            "physicalName": top1.candidate_physical,
            "termNameKo": top1.candidate_term_name_ko,
            "termUri": str(top1.term_uri),
            "valueType": top1.value_type,
        }
    }]

def interactive_confirm(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple CLI confirmation. Returns concept->selected option mapping.
    """
    confirmed: Dict[str, Any] = {}
    for cp in checkpoints:
        concept = cp["concept"]
        print("\n" + "=" * 80)
        print(cp["question"])
        print("-" * 80)
        opts = cp.get("options", [])
        for i, o in enumerate(opts, start=1):
            print(f"{i}. {o['termNameKo']} ({o['physicalName']}) [{o['valueType']}] score={o['score']:.3f}")
        rec = cp.get("recommended", {})
        print(f"\n추천: {rec.get('termNameKo')} ({rec.get('physicalName')}) [{rec.get('valueType')}]")
        while True:
            ans = input(f"선택하세요 (1-{len(opts)}), Enter=추천 사용: ").strip()
            if ans == "":
                chosen = rec
                break
            if ans.isdigit() and 1 <= int(ans) <= len(opts):
                chosen = {
                    "physicalName": opts[int(ans)-1]["physicalName"],
                    "termNameKo": opts[int(ans)-1]["termNameKo"],
                    "termUri": opts[int(ans)-1]["termUri"],
                    "valueType": opts[int(ans)-1]["valueType"],
                }
                break
            print("잘못된 입력입니다. 다시 선택하세요.")
        confirmed[concept] = chosen
    return confirmed

def apply_sql_policy_auto_select(concept: str, ranked: List[CandidateScore]) -> Optional[CandidateScore]:
    """
    Auto-select with hard SQL policy tie-breaker:
    - If SQL_DEFAULT: prefer CODE/ID/NO over NAME when top scores are close.
    - Otherwise keep top1.
    """
    if not ranked:
        return None
    if not sql_policy_prefers_code(concept):
        return ranked[0]

    # If top1 is NAME but there exists CODE/ID/NO within small margin, switch.
    top1 = ranked[0]
    if top1.value_type != "NAME":
        return top1

    for cs in ranked[1:5]:
        if cs.value_type in ("CODE","ID","NO"):
            if (top1.rule_score - cs.rule_score) <= 0.35:
                return cs
    return top1


def apply_specificity_policy(concept: str, ranked: List[CandidateScore]) -> Optional[CandidateScore]:
    """
    Specificity policy (token-based; NO keyword lists):

    If top1 is SPECIALIZED and TTL provides contains_generic_tokens (e.g., AC/STAT/TCD),
    and there exists a GENERIC candidate whose physicalName token-set covers those tokens,
    then prefer that GENERIC candidate when score margin is small.

    This fixes cases where GENERIC physicalName is composite like AC_STAT_TCD
    (not equal to single token 'AC').
    """
    if not ranked:
        return None

    top1 = ranked[0]
    if top1.specificity != "SPECIALIZED":
        return top1

    # Build best GENERIC candidate per physical token-set coverage
    generic_candidates: List[CandidateScore] = []
    for cs in ranked:
        if cs.specificity == "GENERIC":
            generic_candidates.append(cs)

    if not generic_candidates:
        return top1

    # Tokens that define the base concept (from TTL)
    need = {norm_text(t).upper() for t in (top1.contains_generic_tokens or []) if norm_text(t)}
    if not need:
        return top1  # nothing to validate against

    MARGIN = 0.50  # only flip when close (avoid over-correcting)

    # Find best GENERIC whose physical tokens cover the needed tokens
    best_generic: Optional[CandidateScore] = None
    for g in generic_candidates:
        g_tokens = phys_tokens(g.candidate_physical)
        if need.issubset(g_tokens):
            if (best_generic is None) or (g.rule_score > best_generic.rule_score):
                best_generic = g

    if best_generic is None:
        return top1

    if (top1.rule_score - best_generic.rule_score) <= MARGIN:
        return best_generic

    return top1

# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(
    benchmark_json: str,
    ttl_path: str,
    out_json: str,
    base_iri: str,
    mode: str = "checkpoint",   # checkpoint | interactive | auto
) -> Dict[str, Any]:
    
    
    #if not os.path.exists(benchmark_json):
    #    raise FileNotFoundError(f"benchmark not found: {benchmark_json}")

    #with open(benchmark_json, "r", encoding="utf-8") as f:
    #    bench = json.load(f)
    bench = benchmark_json
    if isinstance(bench, str):
        with open(bench, "r", encoding="utf-8") as f:
            bench = json.load(f)

    mappings = bench.get("mappings", [])
    if not isinstance(mappings, list):
        raise ValueError("benchmark json: 'mappings' must be a list")

    g = load_graph(ttl_path)
    EX = Namespace(base_iri.rstrip("#") + "#")
    term_index = build_term_index(g, EX)
    label_index = build_label_hash_index(term_index)

    results: List[Dict[str, Any]] = []

    for m in mappings:
        column_candidate = norm_text(m.get("column_candidate", ""))
        embed_cands = m.get("candidates", []) or []
        if not column_candidate:
            continue

        onto_cands = ontology_anchor_candidates(column_candidate, label_index)
        merged = merge_candidates(embed_cands, onto_cands)

        scored: List[CandidateScore] = []
        for c in merged:
            source = norm_text(c.get("source", "embedding"))
            stage = norm_text(c.get("_stage", ""))

            phys = norm_text(c.get("물리명", ""))
            name = norm_text(c.get("용어명", ""))

            term_uri: Optional[URIRef] = None
            if c.get("_term_uri"):
                term_uri = URIRef(norm_text(c["_term_uri"]))
            else:
                term_uri = find_term_uri_by_physical(g, phys)

            if term_uri is None:
                scored.append(CandidateScore(
                    column_candidate=column_candidate,
                    source=source,
                    stage=stage,
                    candidate_term_name_ko=name,
                    candidate_physical=phys,
                    term_uri="",
                    match_trace={"ttl_missing": True},
                    rule_score=-9.0,
                    rule_signals={"ttl_missing": True, "stage": stage, "candidate_physical": phys, "candidate_name": name},
                    ttl_evidence={"ttl_missing": True},
                    specificity="UNKNOWN",
                    contains_generic_tokens=[],
                    value_type=infer_value_type(phys),
                ))
                continue

            prof = load_term_profile(g, term_uri, EX)
            s, sig, trace = rule_signals(column_candidate, c, prof)
            ev = build_evidence_dict(prof)

            final_name = name or safe_first(prof.pref_labels_ko) or safe_first(prof.alt_labels_ko)
            final_phys = phys or safe_first(prof.notation) or safe_first(prof.hidden_labels)
            scored.append(CandidateScore(
                column_candidate=column_candidate,
                source=source,
                stage=stage,
                candidate_term_name_ko=final_name,
                candidate_physical=final_phys,
                term_uri=str(term_uri),
                match_trace=trace,
                rule_score=s,
                rule_signals=sig,
                ttl_evidence=ev,
                specificity=(prof.specificity or 'UNKNOWN'),
                contains_generic_tokens=list(prof.contains_generic_tokens or []),
                value_type=infer_value_type(final_phys),
            ))

        ranked = sorted(scored, key=lambda z: z.rule_score, reverse=True)

        checkpoints = build_semantic_checkpoints(column_candidate, ranked, max_options=4)

        confirmed_links = {}
        if mode == "interactive" and checkpoints:
            confirmed_links = interactive_confirm(checkpoints)

        # final selection: if confirmed for this concept, use it; else auto-policy
        final_selected = None
        decision_type = "AUTO"
        if confirmed_links.get(column_candidate):
            final_selected = confirmed_links[column_candidate]
            decision_type = "CONFIRMED_BY_USER"
        else:
            # 1) SQL policy (CODE/ID/NO preference)
            auto_cs = apply_sql_policy_auto_select(column_candidate, ranked)
            # 2) Specificity policy (GENERIC vs SPECIALIZED)
            if auto_cs is not None:
                ranked_view = [auto_cs] + [x for x in ranked if x is not auto_cs]
                auto_cs = apply_specificity_policy(column_candidate, ranked_view)
            if auto_cs is not None:
                final_selected = {
                    "physicalName": auto_cs.candidate_physical,
                    "termNameKo": auto_cs.candidate_term_name_ko,
                    "termUri": str(auto_cs.term_uri),
                    "valueType": auto_cs.value_type,
                    "score": auto_cs.rule_score,
                }
            decision_type = "AUTO" if mode != "checkpoint" else ("PENDING_CHECKPOINT" if checkpoints else "AUTO")

        emb_cnt = sum(1 for c in merged if norm_text(c.get("source")) == "embedding")
        onto_cnt = sum(1 for c in merged if norm_text(c.get("source")) == "ontology")

        results.append({
            "column_candidate": column_candidate,
            "candidate_pool_size": len(merged),
            "candidate_pool_sources": {"embedding_count": emb_cnt, "ontology_count": onto_cnt},
            "ranked_candidates": [asdict(x) for x in ranked],
            "semantic_checkpoints": checkpoints,
            "final_selection": {
                "decisionType": decision_type,
                "selected": final_selected
            }
        })

    out = {"results": results}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out

def main_ontology_search_selection(BENCHMARK_JSON, lTtl_path):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #ttl_path = os.path.join(BASE_DIR, "ttl", "financial_terms.ttl")
    BASE_IRI = "http://kpmglcnc.digital/finance/meta#"
    
    #BENCHMARK_JSON = os.path.join(BASE_DIR, "interim_result", "3_benchmark.json")


    
    OUT_JSON = os.path.join(BASE_DIR, "interim_result", "4_selection_results.json")
    
    MODE = "checkpoint"

    OUT_JSON= run_pipeline(
        benchmark_json=BENCHMARK_JSON,
        ttl_path=lTtl_path,
        out_json=OUT_JSON,
        base_iri=BASE_IRI,
        mode=MODE,
    )
    #print(f"Done. Wrote: {OUT_JSON}")
    return OUT_JSON
    

#if __name__ == "__main__":
#    main()
