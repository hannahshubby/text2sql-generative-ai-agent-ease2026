from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any, Deque
from collections import defaultdict, deque
import json
from pathlib import Path

def fold(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "")

@dataclass(frozen=True)
class JoinEdge:
    frm: str
    to: str
    frm_col: str
    to_col: str
    is_implicit: bool = False
    def as_dict(self) -> Dict[str, Any]:
        return {
            "from": self.frm, 
            "to": self.to, 
            "from_col": self.frm_col, 
            "to_col": self.to_col,
            "type": "implicit" if self.is_implicit else "explicit"
        }

def bfs_path(adj: Dict[str, List[JoinEdge]], table_pks: Dict[str, Set[str]], start: str, goal: str) -> Optional[List[JoinEdge]]:
    """
    RDB NORMALIZED BFS:
    Enforces the point 3 rule: "The target's join column MUST be its Primary Key (PK)."
    """
    if start == goal: return []
    q: Deque[str] = deque([start])
    visited = {start}
    parent = {}
    
    while q:
        cur = q.popleft()
        # Sort edges: Implicit edges (direct PK matches) first, then explicit
        sorted_edges = sorted(adj.get(cur, []), key=lambda e: not e.is_implicit)
        
        for edge in sorted_edges:
            if edge.to not in visited:
                target_pks = table_pks.get(edge.to, set())
                # Point 3 Violation Check: Target join column must be a PK
                if edge.to_col not in target_pks:
                    continue 
                
                visited.add(edge.to)
                parent[edge.to] = edge
                if edge.to == goal:
                    res = []
                    curr = goal
                    while curr != start:
                        e = parent[curr]
                        res.append(e)
                        curr = e.frm
                    return res[::-1]
                q.append(edge.to)
    return None

def build_join_plan(step4_result: Dict[str, Any], hierarchy: Dict[str, Any], join_graph: Dict[str, Any], col_to_tabs: Dict[str, List[str]]):
    confirmed = step4_result.get("confirmed", {})
    terms = confirmed.get("terms", [])
    
    anchors = hierarchy.get("anchors", {})
    attrs = hierarchy.get("attributes", {})
    tiers = hierarchy.get("table_tiers", {})

    # 1. Build Base Graph & Extract PKs from explicit references
    table_pks = defaultdict(set)
    adj = defaultdict(list)
    for src, rels in join_graph.items():
        for r in rels:
            s_t, t_t = fold(src), fold(r["to"])
            s_c, t_c = fold(r["from_column"]), fold(r["to_column"])
            table_pks[t_t].add(t_c) # Target of a reference is a PK
            adj[s_t].append(JoinEdge(s_t, t_t, s_c, t_c))
            adj[t_t].append(JoinEdge(t_t, s_t, t_c, s_c))

    # 2. PROACTIVE EXPANSION (Implicit Joins)
    # If table A has column X, and X is a PK of Master B, allow direct join.
    for master_t, pks in list(table_pks.items()):
        if tiers.get(master_t) != 1: continue # Only proactive for Masters
        for pk_col in pks:
            source_candidates = col_to_tabs.get(pk_col, [])
            for s_t_raw in source_candidates:
                if isinstance(s_t_raw, dict):
                    s_t = fold(s_t_raw.get("table", ""))
                else:
                    s_t = fold(str(s_t_raw))
                
                if s_t and s_t != master_t:
                    # Add implicit direct normalization path
                    edge = JoinEdge(s_t, master_t, pk_col, pk_col, is_implicit=True)
                    # Avoid duplicates
                    if not any(e.to == master_t and e.to_col == pk_col for e in adj[s_t]):
                        adj[s_t].append(edge)
                        adj[master_t].append(JoinEdge(master_t, s_t, pk_col, pk_col, is_implicit=True))

    target_tables = set()
    mandatory_anchors = set()
    column_bindings = {}

    # 3. Identify targets and force Masters if PK is used
    for t in terms:
        logical, physical = t.get("originalTerm", "").lower(), fold(t.get("physicalName", ""))
        
        # A. Logical Entity Mapping
        if logical in anchors:
            anc_t = anchors[logical]
            target_tables.add(anc_t)
            mandatory_anchors.add(anc_t)
            
        # B. Physical Attribute Mapping
        if physical in attrs:
            owner_t = attrs[physical]
            target_tables.add(owner_t)
            column_bindings[physical] = owner_t

            # [Proactive Rule] If this attribute is a PK of ANY Master table, force that Master to join
            for m_t, m_pks in table_pks.items():
                if physical in m_pks and tiers.get(m_t) == 1:
                    target_tables.add(m_t)
                    mandatory_anchors.add(m_t)

    if not target_tables: return {"error": "No tables detected."}

    # 4. Root Selection (Highest Authority Master)
    master_targets = [t for t in target_tables if tiers.get(t) == 1]
    root = sorted(master_targets or list(target_tables), key=lambda x: x)[0]

    # 5. BFS Connection with PK Integrity
    final_joins = []
    connected = {root}
    errors = []
    # Process Masters first to stabilize the core structure
    sorted_targets = sorted(list(target_tables), key=lambda t: tiers.get(t, 2))

    for target in sorted_targets:
        if target in connected: continue
        best_path = None
        for start_node in list(connected):
            path = bfs_path(adj, table_pks, start_node, target)
            if path:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
        
        if best_path:
            for e in best_path:
                if e.to not in connected:
                    final_joins.append(e.as_dict())
                    connected.add(e.to)
        else:
            errors.append(f"Incomplete Join: {target} cannot be reached via PK-integrity path.")

    return {
        "root_anchor": root,
        "selected_tables": [{"table": t, "tier": tiers.get(t, 2), "mandatory": t in mandatory_anchors} for t in connected],
        "joins": final_joins,
        "column_bindings": column_bindings,
        "diagnostics": {"errors": errors}
    }

def main(step4_result, col2table, jgraph, hierarchy):
    return build_join_plan(step4_result, hierarchy, jgraph, col2table)
