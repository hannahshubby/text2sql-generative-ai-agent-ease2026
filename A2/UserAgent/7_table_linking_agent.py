# -*- coding: utf-8 -*-
"""
Agent2 (Rebuild v3): Table Linking + Join Planning using provided ground-truth JSONs

왜 v3가 맞나?
- 당신이 이미 "원본"에서 만들어둔 2개 파일이 곧 정답 기반 인덱스/그래프입니다.
  - col_to_tables.json : column -> [{schema, table}, ...]
  - join_graph.json    : table -> [{to, from_column, to_column, direction}, ...]

따라서:
- TTL 파싱/캐시(schema_index.json) 불필요
- shared-key heuristic로 join 추론 불필요
- Agent2는 이 두 파일을 사용해 "후보 생성→스코어링→조인경로→column binding"을 수행하면 됩니다.

고정 경로(요구사항: argparse X)
BASE_DIR/
  temp_artifacts/6_column_finalizer_agent.json  (input; selected_columns)
  col_to_tables.json                             (input)
  join_graph.json                                (input)
  temp_artifacts/7_table_join_plan.json          (output)
"""

from __future__ import annotations

import os, re, json, math
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, deque

# -----------------------------
# Fixed Paths (NO argparse)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_JSON = ""
COL_TO_TABLES_JSON = os.path.join(BASE_DIR, "ttl", "col_to_tables.json")
JOIN_GRAPH_JSON = os.path.join(BASE_DIR, "ttl", "join_graph.json")

OUT_JSON = ""

# -----------------------------
# Internal constants (NO external inputs)
# -----------------------------
TOP_TABLE_CANDIDATES = 12          # 테이블 후보는 너무 많으면 조인 탐색 폭발
TOP_JOIN_PLAN_CANDIDATES = 5       # 사람이 리뷰 가능한 수준
MAX_JOIN_HOPS = 5                  # join_graph가 풍부하면 4~6 사이가 실무적으로 무난
MAX_COMBO_SETS = 120               # 후보 조합 탐색 상한

# Scoring weights (조정 가능)
W_COVERAGE = 2.0
W_JOIN_COST = -0.35
W_PREF = 0.5

# -----------------------------
# Utilities
# -----------------------------
def norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()

def fold(s: str) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def table_preference_score(schema: str, table: str) -> float:
    """
    조직 정책 반영 포인트.
    - 예: public 스키마(뷰/머트뷰/스테이징) 패널티 등
    지금은 최소한으로만 둠.
    """
    sc = fold(schema)
    t = fold(table)

    score = 0.0
    # 예시 정책: public은 레퍼런스/테스트가 섞일 수 있어 약한 페널티
    if sc == "public":
        score -= 0.03

    # 예시 정책: view/agg/hist 패턴 페널티(테이블명이 코드형이면 영향 없음)
    if t.startswith("vw") or "view" in t:
        score -= 0.10
    if "hist" in t or "log" in t or "audit" in t:
        score -= 0.05
    if "agg" in t or "sum" in t:
        score -= 0.05

    return score

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Load inputs
# -----------------------------
def read_selected_columns(path: str) -> List[Dict[str, Any]]:
    obj = read_json(path)
    return obj.get("selected_columns") or obj.get("final_selected_columns") or []

def load_col_to_tables(path: str) -> Dict[str, List[Dict[str, str]]]:
    # column (lower/whatever) -> [{schema, table}, ...]
    return read_json(path)

def load_join_graph(path: str) -> Dict[str, List[Dict[str, Any]]]:
    # table -> [{to, from_column, to_column, direction}, ...]
    return read_json(path)

# -----------------------------
# Step 2-1: Candidate Generation (col_to_tables 기반)
# -----------------------------
def build_column_table_candidates(selected_columns: List[Dict[str, Any]], col_to_tables: Dict[str, Any]) -> Dict[str, Any]:
    out_cols = []
    pool = set()

    for sc in selected_columns:
        phys = norm_text(sc.get("physicalName"))
        col_only = phys.split(".")[-1]
        key = fold(col_only)

        candidates = col_to_tables.get(key) or col_to_tables.get(col_only) or []
        # normalize
        norm_cands = []
        for c in candidates:
            schema = norm_text(c.get("schema"))
            table = norm_text(c.get("table"))
            if not table:
                continue
            norm_cands.append({"schema": schema or "", "table": table, "evidence": "col_to_tables"})
            pool.add((schema or "", table))

        out_cols.append({
            "concept": norm_text(sc.get("concept") or phys),
            "physicalName": phys,
            "column": col_only,
            "candidateTables": norm_cands,
        })

    return {
        "column_table_candidates": out_cols,
        "table_pool": [{"schema": s, "table": t} for (s, t) in sorted(pool)]
    }

# -----------------------------
# Step 2-2: Table Scoring (coverage + preference)
# -----------------------------
def score_table_candidates(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    per_col = cand["column_table_candidates"]
    total_cols = max(1, len(per_col))

    cov = defaultdict(int)
    schema_of = {}  # (schema, table) key 유지
    ev = defaultdict(list)

    for c in per_col:
        phys = c["physicalName"]
        for t in c["candidateTables"]:
            schema = t["schema"]
            table = t["table"]
            k = (schema, table)
            cov[k] += 1
            schema_of[k] = schema
            ev[k].append(f"contains:{phys}")

    scored = []
    for (schema, table), cnt in cov.items():
        coverage_ratio = cnt / total_cols
        pref = table_preference_score(schema, table)
        raw = coverage_ratio + pref
        scored.append({
            "schema": schema,
            "table": table,
            "coverage_cols": cnt,
            "coverage_ratio": coverage_ratio,
            "pref_score": pref,
            "raw_score": raw,
            "evidence": ev[(schema, table)],
        })

    scored.sort(key=lambda x: (x["raw_score"], x["coverage_cols"]), reverse=True)
    return scored[:max(3, TOP_TABLE_CANDIDATES)]

# -----------------------------
# Step 2-3/2-4: Join path search on join_graph (BFS by hops, tie-break by edge count)
# -----------------------------
def bfs_join_path(join_graph: Dict[str, List[Dict[str, Any]]], start: str, goal: str, max_hops: int) -> Optional[List[Dict[str, Any]]]:
    if start == goal:
        return []

    q = deque([start])
    prev = {}  # node -> (parent, edge_dict)
    depth = {start: 0}

    while q:
        cur = q.popleft()
        if depth[cur] >= max_hops:
            continue
        for e in join_graph.get(cur, []):
            nxt = e.get("to")
            if not nxt:
                continue
            if nxt not in depth:
                depth[nxt] = depth[cur] + 1
                prev[nxt] = (cur, e)
                q.append(nxt)

    if goal not in prev:
        return None

    # reconstruct
    path = []
    node = goal
    while node != start:
        parent, edge = prev[node]
        path.append((parent, node, edge))
        node = parent
    path.reverse()

    # normalize as join_edges list (left=parent, right=node)
    out = []
    for left, right, e in path:
        out.append({
            "left": left,
            "right": right,
            "on": [{"leftKey": e.get("from_column"), "rightKey": e.get("to_column")}],
            "joinType": "INNER",
            "reason": "join_graph",
            "direction": e.get("direction"),
        })
    return out

def tables_for_column(col_only: str, col_to_tables: Dict[str, Any]) -> List[Tuple[str, str]]:
    key = fold(col_only)
    candidates = col_to_tables.get(key) or col_to_tables.get(col_only) or []
    out = []
    for c in candidates:
        schema = norm_text(c.get("schema"))
        table = norm_text(c.get("table"))
        if table:
            out.append((schema or "", table))
    return out

def choose_core_table(table_cands: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not table_cands:
        return ("", "")
    return (table_cands[0].get("schema") or "", table_cands[0].get("table") or "")

def join_cost(join_edges: List[Dict[str, Any]]) -> float:
    # 단순: hop 수 기반 비용
    return float(len(join_edges))

def build_join_plan_candidates(
    selected_columns: List[Dict[str, Any]],
    col_to_tables: Dict[str, Any],
    join_graph: Dict[str, Any],
    table_cands: List[Dict[str, Any]],
) -> Dict[str, Any]:
    core_schema, core_table = choose_core_table(table_cands)
    core = core_table

    # 후보 테이블 set: TOP 테이블 후보 기반
    cand_tables = [(tc["schema"], tc["table"]) for tc in table_cands]
    cand_table_names = {t for _, t in cand_tables}  # join_graph는 table명만 키로 사용

    # 컬럼별 테이블 옵션(후보군에 속하는 것만 최대 2개)
    per_col_opts = []
    for sc in selected_columns:
        phys = norm_text(sc.get("physicalName"))
        col_only = phys.split(".")[-1]
        opts = [t for (s, t) in tables_for_column(col_only, col_to_tables) if t in cand_table_names][:2]
        per_col_opts.append((phys, col_only, opts))

    # 제한된 조합 생성(폭발 방지)
    combo_sets = set()
    combo_sets.add(tuple(sorted([core] if core else [])))

    for _, _, opts in per_col_opts:
        if not opts:
            continue
        new_sets = set()
        for s in list(combo_sets):
            for t in opts:
                new_sets.add(tuple(sorted(set(s) | {t})))
        combo_sets |= new_sets
        if len(combo_sets) > MAX_COMBO_SETS:
            break

    plans = []
    for ts in list(combo_sets):
        tables = sorted(set([t for t in ts if t]))
        if core and core not in tables:
            tables.append(core)
            tables = sorted(set(tables))
        if not tables:
            continue

        # join edges to connect all tables from core
        edges = []
        connected = {core} if core else {tables[0]}

        ok = True
        for t in tables:
            if t in connected:
                continue
            # connect from any connected node
            path = None
            for c in list(connected):
                path = bfs_join_path(join_graph, c, t, MAX_JOIN_HOPS)
                if path is not None:
                    break
            if path is None:
                ok = False
                break
            # add path edges
            for pe in path:
                edges.append(pe)
                connected.add(pe["right"])

        if not ok:
            continue

        # dedup edges by (left,right,on)
        seen = set()
        dedup = []
        for e in edges:
            k = (e["left"], e["right"], e["on"][0]["leftKey"], e["on"][0]["rightKey"])
            if k in seen:
                continue
            seen.add(k)
            dedup.append(e)
        edges = dedup

        # coverage: selected columns that are present in any chosen table
        cover = 0
        for _, col_only, _ in per_col_opts:
            tabs = {t for _, t in tables_for_column(col_only, col_to_tables)}
            if any(t in tabs for t in tables):
                cover += 1
        coverage_ratio = cover / max(1, len(selected_columns))

        cost = join_cost(edges)

        # preference: schema/table preference (schema는 core 후보에서만 신뢰; 여기서는 table명만 남는 한계)
        # -> table_cands에서 schema 정보를 최대한 끌어옴
        schema_map = {(tc["table"]): (tc.get("schema") or "") for tc in table_cands}
        pref = sum(table_preference_score(schema_map.get(t, ""), t) for t in tables)

        total = (W_COVERAGE * coverage_ratio) + (W_JOIN_COST * cost) + (W_PREF * pref)

        # column bindings: 각 selected column을 어떤 table에 바인딩할지
        bindings = []
        for sc in selected_columns:
            phys = norm_text(sc.get("physicalName"))
            col_only = phys.split(".")[-1]
            tabs = tables_for_column(col_only, col_to_tables)
            tab_names = [t for _, t in tabs]
            chosen = core if core in tables and core in tab_names else None
            if not chosen:
                for t in tables:
                    if t in tab_names:
                        chosen = t
                        break
            if not chosen and tab_names:
                chosen = tab_names[0]

            bindings.append({
                "concept": norm_text(sc.get("concept") or phys),
                "physicalName": phys,
                "column": col_only,
                "boundTable": chosen or "",
                "termUri": norm_text(sc.get("termUri")),
                "confidence": float(sc.get("confidence") or 0.0),
                "needsHumanReview": bool(sc.get("needsHumanReview") or False),
                "source": norm_text(sc.get("source")),
            })

        plans.append({
            "tables": tables,
            "join_edges": edges,
            "column_bindings": bindings,
            "coverage_ratio": coverage_ratio,
            "join_cost": cost,
            "pref_score": pref,
            "total_score": total,
            "trace": {"core_table": core, "coverage_count": cover, "n_tables": len(tables), "n_join_edges": len(edges)},
        })

    plans.sort(key=lambda p: (p["total_score"], p["coverage_ratio"], -p["join_cost"]), reverse=True)
    plans = plans[:max(1, TOP_JOIN_PLAN_CANDIDATES)]

    return {
        "join_plan_candidates": plans,
        "selected_plan": plans[0] if plans else {},
        "trace": {"core_table": core, "core_schema": core_schema}
    }

# -----------------------------
# Run
# -----------------------------
def run(in_json: str, out_json: str, col_to_tables_json: str | None = None, join_graph_json: str | None = None):
    IN_JSON = in_json
    OUT_JSON = out_json
    global COL_TO_TABLES_JSON, JOIN_GRAPH_JSON
    if col_to_tables_json is not None:
        COL_TO_TABLES_JSON = col_to_tables_json
    if join_graph_json is not None:
        JOIN_GRAPH_JSON = join_graph_json
    for p in (IN_JSON, COL_TO_TABLES_JSON, JOIN_GRAPH_JSON):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    selected_columns = read_selected_columns(IN_JSON)
    col_to_tables = load_col_to_tables(COL_TO_TABLES_JSON)
    join_graph = load_join_graph(JOIN_GRAPH_JSON)

    # 2-1
    cand = build_column_table_candidates(selected_columns, col_to_tables)
    # 2-2
    table_cands = score_table_candidates(cand)
    # 2-4/2-5
    plans = build_join_plan_candidates(selected_columns, col_to_tables, join_graph, table_cands)

    out = {
        "table_linking": {
            "column_table_candidates": cand["column_table_candidates"],
            "table_candidates": table_cands,
        },
        "join_planning": {
            "join_plan_candidates": plans["join_plan_candidates"],
            "selected_plan": plans["selected_plan"],
            "trace": plans["trace"],
        },
        "meta": {
            "inputs": {
                "selected_columns": os.path.basename(IN_JSON),
                "col_to_tables": os.path.basename(COL_TO_TABLES_JSON),
                "join_graph": os.path.basename(JOIN_GRAPH_JSON),
            },
            "constants": {
                "TOP_TABLE_CANDIDATES": TOP_TABLE_CANDIDATES,
                "TOP_JOIN_PLAN_CANDIDATES": TOP_JOIN_PLAN_CANDIDATES,
                "MAX_JOIN_HOPS": MAX_JOIN_HOPS,
                "MAX_COMBO_SETS": MAX_COMBO_SETS,
                "W_COVERAGE": W_COVERAGE,
                "W_JOIN_COST": W_JOIN_COST,
                "W_PREF": W_PREF,
            }
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Agent2 wrote: {OUT_JSON}")
    sp = out["join_planning"].get("selected_plan") or {}
    if sp:
        print("  selected tables:", sp.get("tables"))
        print("  coverage_ratio:", round(sp.get("coverage_ratio", 0), 3),
              "join_cost:", sp.get("join_cost"),
              "total_score:", round(sp.get("total_score", 0), 3))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python 7_table_linking_agent.py <in_json> <out_json> [col_to_tables_json] [join_graph_json]")
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None, sys.argv[4] if len(sys.argv)>4 else None)



# -----------------------------------------------------------------------------
# In-memory API (no file artifacts required)
# -----------------------------------------------------------------------------
def run_in_memory(selected_columns_obj: Dict[str, Any], col_to_tables: Dict[str, Any], join_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Pure in-memory entrypoint for Agent2.

    Returns the SAME schema as legacy file-based `run(...)`:
    {
      "table_linking": {...},
      "join_planning": {...},
      "meta": {...}
    }
    """
    selected_columns = (
        selected_columns_obj.get("selected_columns")
        or selected_columns_obj.get("final_selected_columns")
        or []
    )
    if not isinstance(selected_columns, list):
        raise TypeError(f"Agent2: expected list for selected_columns, got {type(selected_columns)}")

    # 2-1
    cand = build_column_table_candidates(selected_columns, col_to_tables)
    # 2-2
    table_cands = score_table_candidates(cand)
    # 2-4/2-5
    plans = build_join_plan_candidates(selected_columns, col_to_tables, join_graph, table_cands)

    selected_plan = (plans.get("selected_plan") or {})
    # Fallback: if no join plan could be built, at least pick the top table candidate as a 1-table plan
    if not (selected_plan.get("tables") or []):
        if table_cands:
            top = table_cands[0]
            selected_plan = {"tables": [top["table"]], "join_edges": []}
        else:
            selected_plan = {"tables": [], "join_edges": []}

    out = {
        "table_linking": {
            "column_table_candidates": cand.get("column_table_candidates", []),
            "table_candidates": table_cands,
        },
        "join_planning": {
            "join_plan_candidates": plans.get("join_plan_candidates", []),
            "selected_plan": selected_plan,
            "trace": plans.get("trace", {}),
        },
        "meta": {
            "inputs": {"selected_columns": "in_memory"},
        },
    }
    return out


# -----------------------------------------------------------------------------
# Backward-compatible file entrypoint
# -----------------------------------------------------------------------------
def run() -> None:
    if not IN_JSON or not OUT_JSON:
        raise RuntimeError("IN_JSON/OUT_JSON must be set")
    sel = read_json(IN_JSON)
    col_to_tables = load_col_to_tables(COL_TO_TABLES_JSON)
    join_graph = load_join_graph(JOIN_GRAPH_JSON)
    out = run_in_memory(sel, col_to_tables, join_graph)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
