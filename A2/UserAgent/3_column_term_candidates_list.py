
"""
4_column_term_candidates_list_v2.py

Goal
- Fix "exact same term missing" and improve candidate recall/precision for Korean term-dictionary mapping.
- Retrieval pipeline (recent IR best-practice style):
  0) Normalise + exact lookup (must-include)
  1) Prefix/substring lexical match (must-include, capped)
  2) Fuzzy/char-ngram rerank (RapidFuzz) (capped)
  3) Hybrid retrieval (BM25 + dense FAISS) with wider recall pool
  4) Fusion + stage-priority rerank, then topN

Notes
- This script is designed to be a drop-in replacement for TermMapper.map_candidate_topk().
- It keeps your output JSON layout:
    {"column_candidate": "...", "candidates": [{"용어명":..., "물리명":..., ...}, ...]}
- It assumes you already have cache_terms/{terms.jsonl, faiss.index} built.
"""

from __future__ import annotations

import os, re, json, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set

import numpy as np
import faiss
from openai import OpenAI
from rapidfuzz import fuzz, process
from rank_bm25 import BM25Okapi

from pathlib import Path

TERMS_JSONL = "terms.jsonl"
FAISS_INDEX = "faiss.index"


# ============================================================
# Data model
# ============================================================

@dataclass
class TermRow:
    term_ko: str
    physical: str = ""
    info_type: str = ""
    code_name: str = ""
    definition: str = ""


def load_terms_jsonl(path: str) -> List[TermRow]:
    out: List[TermRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                TermRow(
                    term_ko=str(obj.get("term_ko") or obj.get("용어명") or ""),
                    physical=str(obj.get("physical") or obj.get("물리명") or ""),
                    info_type=str(obj.get("info_type") or obj.get("인포타입") or ""),
                    code_name=str(obj.get("code_name") or obj.get("코드명") or ""),
                    definition=str(obj.get("definition") or obj.get("정의") or ""),
                )
            )
    return out


# ============================================================
# Normalisation + lexical utilities
# ============================================================

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[·\.\,\(\)\[\]\{\}/\\\-\_\:\;\|]+")

def norm_basic(s: Any) -> str:
    """Normalise for dictionary matching (Korean-friendly).
    - lowercase (for ASCII)
    - remove whitespace
    - unify punctuation separators
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _WS.sub("", s)
    # unify common separators to nothing (for tight key match)
    s = _PUNCT.sub("", s)
    return s

def safe_contains(hay: str, needle: str) -> bool:
    if not needle:
        return False
    return needle in hay

def safe_startswith(hay: str, needle: str) -> bool:
    if not needle:
        return False
    return hay.startswith(needle)

def char_ngrams(s: str, n: int = 3) -> Set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(0, len(s)-n+1)}


# ============================================================
# Candidate fusion helpers
# ============================================================

STAGE_PRIORITY = {
    "exact": 1000,
    "prefix": 900,
    "substring": 850,
    "fuzzy": 700,
    "bm25": 500,
    "vector": 480,
}

# Stage weights: used in final score = priority + fusion_score
STAGE_WEIGHT = {
    "exact": 1.0,
    "prefix": 0.9,
    "substring": 0.85,
    "fuzzy": 0.7,
    "bm25": 0.5,
    "vector": 0.48,
}

def rrf(rank: int, k: int = 60) -> float:
    # reciprocal rank fusion component
    return 1.0 / (k + rank + 1)

@dataclass
class CandHit:
    idx: int
    stage: str
    score: float  # stage-local score (e.g. fuzz ratio, bm25 score, faiss score, etc.)
    rank: int     # rank within stage list (0-based)

def merge_hits(hits: Iterable[CandHit]) -> Dict[int, CandHit]:
    """Merge duplicates by keeping the best stage (highest priority).
    For same stage duplicates, keep higher score (or better rank).
    """
    best: Dict[int, CandHit] = {}
    for h in hits:
        prev = best.get(h.idx)
        if prev is None:
            best[h.idx] = h
            continue

        # Prefer higher stage priority
        if STAGE_PRIORITY[h.stage] > STAGE_PRIORITY[prev.stage]:
            best[h.idx] = h
            continue

        if STAGE_PRIORITY[h.stage] < STAGE_PRIORITY[prev.stage]:
            continue

        # Same stage: prefer higher score, then better rank
        if h.score > prev.score:
            best[h.idx] = h
        elif h.score == prev.score and h.rank < prev.rank:
            best[h.idx] = h
    return best


# ============================================================
# TermMapper v2
# ============================================================

class TermMapperV2:
    def __init__(
        self,
        cache_dir: str = "cache_terms",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
        use_bm25: bool = True,
        fuzzy_scorer: str = "QRatio",  # "QRatio" or "WRatio"
        fuzzy_threshold: int = 60,
        trigram_prefilter: bool = True,
        prefilter_max: int = 4000,
    ):
        self.cache_dir = cache_dir
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key="")

        terms_path = os.path.join(cache_dir, TERMS_JSONL)
        if not os.path.exists(terms_path):
            raise FileNotFoundError(f"terms.jsonl not found: {terms_path} (run build_term_index.py first)")
        self.terms: List[TermRow] = load_terms_jsonl(terms_path)
        if not self.terms:
            raise RuntimeError("Loaded terms are empty. Check terms.jsonl.")

        idx_path = os.path.join(cache_dir, FAISS_INDEX)
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"faiss.index not found: {idx_path} (run build_term_index.py first)")
        self.faiss_index = faiss.read_index(idx_path)

        if self.faiss_index.ntotal != len(self.terms):
            raise RuntimeError(
                f"Index/Terms mismatch: faiss.ntotal={self.faiss_index.ntotal} vs terms={len(self.terms)}. "
                f"Rebuild the cache."
            )

        # Normalised strings
        self.term_norm: List[str] = [norm_basic(t.term_ko) for t in self.terms]

        # Exact index: norm -> list of term idx (some dictionaries may have duplicates)
        self.exact_index: Dict[str, List[int]] = {}
        for i, nk in enumerate(self.term_norm):
            if not nk:
                continue
            self.exact_index.setdefault(nk, []).append(i)

        # Trigram inverted index (for fast prefilter on substring/fuzzy)
        self.trigram_prefilter = trigram_prefilter
        self.prefilter_max = prefilter_max
        self.tri_inv: Dict[str, List[int]] = {}
        if trigram_prefilter:
            for i, nk in enumerate(self.term_norm):
                for g in char_ngrams(nk, 3):
                    self.tri_inv.setdefault(g, []).append(i)

        # BM25
        self.bm25 = None
        if use_bm25:
            corpus = [self._tokenize_for_bm25(t.term_ko + " " + (t.definition or "")) for t in self.terms]
            self.bm25 = BM25Okapi(corpus)

        # Fuzzy scorer
        self.fuzzy_threshold = int(fuzzy_threshold)
        if fuzzy_scorer == "WRatio":
            self._fuzz = fuzz.WRatio
        else:
            self._fuzz = fuzz.QRatio  # good default for Korean strings too

    # ------------------------
    # BM25 utilities
    # ------------------------
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        text = str(text or "")
        # keep Korean blocks and alnum; split for BM25
        # Add 2-gram for Korean to increase recall (your original idea), but keep it bounded.
        tokens = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text.lower())
        out: List[str] = []
        for tok in tokens:
            out.append(tok)
            if re.fullmatch(r"[가-힣]+", tok) and len(tok) >= 2:
                # add 2-grams
                out.extend([tok[i:i+2] for i in range(len(tok)-1)])
        return out

    def bm25_retrieve(self, query: str, topn: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        q_tokens = self._tokenize_for_bm25(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:topn]
        return [(int(i), float(s)) for i, s in ranked if s > 0]

    # ------------------------
    # Vector utilities
    # ------------------------
    def vector_retrieve(self, query: str, topn: int) -> List[Tuple[int, float]]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=[query])
        qv = np.array([resp.data[0].embedding], dtype="float32")
        faiss.normalize_L2(qv)
        scores, idxs = self.faiss_index.search(qv, topn)
        out: List[Tuple[int, float]] = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0:
                continue
            out.append((int(i), float(s)))
        return out

    # ------------------------
    # Prefilter utilities
    # ------------------------
    def _prefilter_indices(self, q_norm: str) -> Optional[List[int]]:
        if not self.trigram_prefilter:
            return None
        grams = list(char_ngrams(q_norm, 3))
        if not grams:
            return None
        # gather counts (simple voting)
        counts: Dict[int, int] = {}
        for g in grams[:50]:  # cap grams to prevent blow-up
            for idx in self.tri_inv.get(g, []):
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return None
        # take top indices by gram-overlap
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: self.prefilter_max]
        return [i for i, _c in ranked]

    # ------------------------
    # Stage retrieval
    # ------------------------
    def _stage_exact(self, q_norm: str, k: int) -> List[CandHit]:
        idxs = self.exact_index.get(q_norm, [])
        hits: List[CandHit] = []
        for r, idx in enumerate(idxs[:k]):
            hits.append(CandHit(idx=idx, stage="exact", score=1.0, rank=r))
        return hits

    def _stage_prefix_substring(self, q_norm: str, k_prefix: int, k_sub: int) -> List[CandHit]:
        # Use prefilter to limit scanning if possible
        cand_idxs = self._prefilter_indices(q_norm)
        idx_iter: Iterable[int] = cand_idxs if cand_idxs is not None else range(len(self.term_norm))

        prefix_hits: List[int] = []
        sub_hits: List[int] = []
        for idx in idx_iter:
            tn = self.term_norm[idx]
            if not tn:
                continue
            if safe_startswith(tn, q_norm):
                prefix_hits.append(idx)
            elif safe_contains(tn, q_norm):
                sub_hits.append(idx)
            # early break if we already have enough and we're scanning full space
            if cand_idxs is None and len(prefix_hits) >= k_prefix * 5 and len(sub_hits) >= k_sub * 5:
                # keep some buffer before sorting
                break

        # Prefer shorter term for prefix/substr (closer to base term), tie-break by length
        prefix_hits = sorted(prefix_hits, key=lambda i: (len(self.term_norm[i]), i))[:k_prefix]
        sub_hits = sorted(sub_hits, key=lambda i: (len(self.term_norm[i]), i))[:k_sub]

        hits: List[CandHit] = []
        for r, idx in enumerate(prefix_hits):
            # score: inverse length to favor concise base terms
            hits.append(CandHit(idx=idx, stage="prefix", score=1.0 / max(1, len(self.term_norm[idx])), rank=r))
        for r, idx in enumerate(sub_hits):
            hits.append(CandHit(idx=idx, stage="substring", score=1.0 / max(1, len(self.term_norm[idx])), rank=r))
        return hits

    def _stage_fuzzy(self, q_norm: str, k: int) -> List[CandHit]:
        # Use prefilter pool for speed and higher quality in big dictionaries.
        cand_idxs = self._prefilter_indices(q_norm)
        if cand_idxs is None:
            choices = self.term_norm
            choice_map = None
        else:
            choices = [self.term_norm[i] for i in cand_idxs]
            choice_map = cand_idxs  # map local position -> original idx

        # RapidFuzz: returns (match, score, idx_in_choices)
        results = process.extract(
            q_norm,
            choices,
            scorer=self._fuzz,
            limit=max(k * 5, k),
        )
        hits: List[CandHit] = []
        for r, (_m, sc, pos) in enumerate(results):
            if sc < self.fuzzy_threshold:
                continue
            idx = pos if choice_map is None else choice_map[pos]
            hits.append(CandHit(idx=int(idx), stage="fuzzy", score=float(sc) / 100.0, rank=r))
            if len(hits) >= k:
                break
        return hits

    def _stage_bm25_vector(self, query: str, k_each: int, recall_pool: int) -> List[CandHit]:
        hits: List[CandHit] = []
        # BM25 recall_pool then keep top k_each
        bm = self.bm25_retrieve(query, topn=recall_pool)
        for r, (idx, sc) in enumerate(bm[:k_each]):
            hits.append(CandHit(idx=idx, stage="bm25", score=sc, rank=r))

        vec = self.vector_retrieve(query, topn=recall_pool)
        for r, (idx, sc) in enumerate(vec[:k_each]):
            hits.append(CandHit(idx=idx, stage="vector", score=sc, rank=r))
        return hits

    # ------------------------
    # Public API
    # ------------------------
    def map_candidate_staged(
        self,
        column_candidate: str,
        topn: int = 10,
        # stage quotas (defaults: 10 each, but exact/prefix usually smaller naturally)
        k_exact: int = 10,
        k_prefix: int = 10,
        k_substring: int = 10,
        k_fuzzy: int = 10,
        k_bm25: int = 10,
        k_vector: int = 10,
        # wider recall pools for bm25/vector to avoid "top10 cut" failure
        recall_pool: Optional[int] = None,
        # optional ontology evidence bonus: fn(term: TermRow, candidate: str) -> float
        evidence_bonus_fn: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        q = str(column_candidate or "")
        q_norm = norm_basic(q)

        # choose recall pool automatically
        if recall_pool is None:
            recall_pool = max(50, topn * 10)

        stage_hits: List[CandHit] = []
        stage_hits.extend(self._stage_exact(q_norm, k_exact))

        stage_hits.extend(self._stage_prefix_substring(q_norm, k_prefix, k_substring))

        stage_hits.extend(self._stage_fuzzy(q_norm, k_fuzzy))

        stage_hits.extend(self._stage_bm25_vector(q, k_each=max(k_bm25, k_vector), recall_pool=recall_pool))

        # Merge duplicates, keep best stage per term
        merged = merge_hits(stage_hits)

        # Fusion score: stage priority + stage weight + RRF by rank within stage
        scored: List[Tuple[float, CandHit]] = []
        for idx, hit in merged.items():
            base = float(STAGE_PRIORITY[hit.stage])
            fusion = STAGE_WEIGHT[hit.stage] + rrf(hit.rank)
            # add optional ontology evidence bonus (lightweight hook)
            if evidence_bonus_fn is not None:
                try:
                    fusion += float(evidence_bonus_fn(self.terms[idx], q))
                except Exception:
                    pass
            scored.append((base + fusion, hit))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build output in your existing format
        out: List[Dict[str, Any]] = []
        for _score, hit in scored[:topn]:
            t = self.terms[hit.idx]
            out.append(
                {
                    "용어명": t.term_ko,
                    "물리명": t.physical,
                    "인포타입": t.info_type,
                    "코드명": t.code_name,
                    "정의": t.definition,
                    "_stage": hit.stage,   # debug field: you can remove later
                }
            )
        return out

    def map_one(self, column_candidate: str, topn: int = 10) -> Dict[str, Any]:
        return {"column_candidate": column_candidate, "candidates": self.map_candidate_staged(column_candidate, topn=topn)}

    def map_many(self, column_candidates: List[str], topn: int = 10) -> Dict[str, Any]:
        mappings = [self.map_one(c, topn=topn) for c in column_candidates]
        return {"mappings": mappings}


# ============================================================
# CLI
# ============================================================

def _load_column_candidates_from_json(path: str) -> List[str]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    # support both {"column_candidates":[...]} and raw list [...]
    if isinstance(obj, dict) and "column_candidates" in obj:
        return list(obj["column_candidates"])
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError("Unsupported input JSON. Provide a list or {'column_candidates':[...]}")

def main_column_term_candidates_list(data, cache_dir=None):
    
    outputPath=r"interim_result\3_benchmark.json"

    column_candidates_list = data["column_candidates"]

    openai_api_key=""

    # Use absolute path if cache_dir is not provided
    if cache_dir is None:
        # Fallback logic if config is not imported, but in our orchestrator we should pass it
        try:
            from config import CACHE_DIR
            cache_dir = str(CACHE_DIR)
        except ImportError:
            cache_dir = "cache_terms"

    mapper = TermMapperV2(
        cache_dir=cache_dir,
        embedding_model="text-embedding-3-small",
        openai_api_key=openai_api_key,
        use_bm25=True,
        fuzzy_scorer="QRatio",
        fuzzy_threshold=60,
        trigram_prefilter=True,
        prefilter_max=4000,
    )

    result = mapper.map_many(column_candidates_list, topn=50)

    return result
    #with open(outputPath, "w", encoding="utf-8") as f:
    #    json.dump(result, f, indent=2, ensure_ascii=False)

    #print(f"Saved: {outputPath} (mappings={len(column_candidates_list)})")


