import os
import json
import sys
from typing import Dict, Any

# Ensure we can import config and other scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import *

import importlib.util

def import_module_from_path(module_name: str, file_name: str):
    file_path = os.path.join(current_dir, file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import specific agents using helper for files with underscores/digits
rewrite_agent = import_module_from_path("rewrite_agent", "1_UserCanonicalRewriteAgent.py")
extract_agent = import_module_from_path("extract_agent", "2_column_candidates_list.py")
term_mapper_agent = import_module_from_path("term_mapper_agent", "3_column_term_candidates_list.py")
ontology_agent = import_module_from_path("ontology_agent", "4_OntologySearchSelection_semantic_checkpoints_specificity_policy_gates.py")
llm_finalize_agent = import_module_from_path("llm_finalize_agent", "5_llm_finalize_using_intent.py")
column_finalizer_agent = import_module_from_path("column_finalizer_agent", "6_column_finalizer_agent_v2.py")
table_linking_agent = import_module_from_path("table_linking_agent", "7_table_linking_agent.py")
filter_planner_agent = import_module_from_path("filter_planner_agent", "8_filter_planner_agent_v3.py")
code_resolver_agent = import_module_from_path("code_resolver_agent", "8_5_code_value_resolver_agent_v7.py")
sql_synthesizer_agent = import_module_from_path("sql_synthesizer_agent", "9_sql_synthesizer_agent_v2.py")

def run_pipeline(user_query: str) -> str:
    # Ensure cache exists (minimal check)
    terms_jsonl = CACHE_DIR / "terms.jsonl"
    if not terms_jsonl.exists():
        print(f"Warning: {terms_jsonl} not found. Some agents might fail. Please ensure terms index is built.")

    print(f"--- Step 1: Canonical Rewrite ---")
    try:
        final_rewrite = rewrite_agent.main_UserCanonicalRewriteAgent(user_query, interactive=False)
    except Exception as e:
        print(f" [!] Step 1 Failed: {e}")
        final_rewrite = user_query
    
    print(f"--- Step 2: Extraction ---")
    try:
        extract_result = extract_agent.main_extract_column_candidates(final_rewrite)
    except Exception as e:
        print(f" [!] Step 2 Failed: {e}")
        extract_result = {"user_query": user_query, "column_candidates": []}
    
    print(f"--- Step 3: Term Mapping ---")
    try:
        # Pass cache_dir to Agent 3
        benchmark_data = term_mapper_agent.main_column_term_candidates_list(extract_result, cache_dir=str(CACHE_DIR))
    except Exception as e:
        print(f" [!] Step 3 Failed: {e}")
        benchmark_data = {"termMentions": []}
    
    print(f"--- Step 4: Ontology Selection ---")
    # This expects a path to a file, but we can pass local data if we modify it or just use interim files
    try:
        benchmark_path = INTERIM_DIR / "3_benchmark.json"
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
        
        selection_results = ontology_agent.main_ontology_search_selection(benchmark_data, str(ONTOLOGY_TTL))
    except Exception as e:
        print(f" [!] Step 4 Failed: {e}")
        selection_results = {"search_results": []}
    
    print(f"--- Step 5: LLM Finalize ---")
    try:
        final_selection_llm = llm_finalize_agent.main_llm_finalize_using_intent(final_rewrite, selection_results)
    except Exception as e:
        print(f" [!] Step 5 Failed: {e}")
        final_selection_llm = {"final_selected_columns": []}
    
    print(f"--- Step 6: Column Finalizer ---")
    try:
        selected_columns = column_finalizer_agent.finalize_columns(final_selection_llm)
    except Exception as e:
        print(f" [!] Step 6 Failed: {e}")
        selected_columns = {"selected_columns": []}
    
    print(f"--- Step 7: Table Linking ---")
    try:
        # Load required data for table linking
        with open(COL_TO_TABLES_JSON, "r", encoding="utf-8") as f:
            col_to_tables = json.load(f)
        with open(JOIN_GRAPH_JSON, "r", encoding="utf-8") as f:
            join_graph = json.load(f)
        
        join_plan_obj = table_linking_agent.run_in_memory(selected_columns, col_to_tables, join_graph)
        
        # Check if valid tables were found
        sp = join_plan_obj.get("join_planning", {}).get("selected_plan", {})
        if not sp.get("tables"):
            raise ValueError("No tables linked")
            
    except Exception as e:
        print(f" [!] Step 7 Failed ({e}). Using DUMMY table structure.")
        join_plan_obj = {
            "join_planning": {
                "selected_plan": {
                    "tables": ["UNKNOWN_TABLE"],
                    "join_edges": [],
                    "column_bindings": []
                }
            },
            "meta": {"fallback": True}
        }
    
    print(f"--- Step 8: Filter Planning ---")
    try:
        constraints_obj = filter_planner_agent.run_in_memory(final_rewrite, join_plan_obj)
    except Exception as e:
        print(f" [!] Step 8 Failed: {e}")
        constraints_obj = {"filters": []}
    
    print(f"--- Step 8.5: Code Resolution ---")
    try:
        with open(CODEBOOK_JSON, "r", encoding="utf-8") as f:
            codebook = json.load(f)
        
        resolved_constraints = code_resolver_agent.run_in_memory(constraints_obj, codebook)
    except Exception as e:
        print(f" [!] Step 8.5 Failed: {e}")
        resolved_constraints = constraints_obj
    
    print(f"--- Step 9: SQL Synthesis ---")
    try:
        sql_result = sql_synthesizer_agent.run_in_memory(join_plan_obj, resolved_constraints)
    except Exception as e:
        print(f" [!] Step 9 Failed: {e}")
        sql_result = {"sql": "-- SQL Synthesis failed internally"}
    
    final_sql = sql_result.get("sql", "Error: SQL not generated")
    print(f"Generated SQL: {final_sql}")
    
    return final_sql

if __name__ == "__main__":
    import pandas as pd
    
    print("=== Text2SQL Agent Wrapper ===")
    print("1. 수기 질의 입력 (Manual Query)")
    print("2. 엑셀 배치 작업 (Excel Batch Process)")
    choice = input("선택하세요 (1 or 2, 기본값 2): ").strip()

    if choice == "1":
        while True:
            query = input("\n질의를 입력하세요 (종료하려면 'exit' 입력): ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                continue
            
            print(f"\n--- Processing: {query} ---")
            try:
                generated_sql = run_pipeline(query)
                print(f"\n[Generated SQL]\n{generated_sql}\n")
            except Exception as e:
                print(f"Error: {e}")
                
    else:
        # 엑셀 파일 경로 설정
        excel_path = BASE_DIR / "fewShotSample_260226.xlsx"
        sheet_name = "FewShotSample"
        
        try:
            print(f"Loading Excel from {excel_path}...")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
            # 결과물 저장을 위해 D열(인덱스 3)이 있는지 확인하고 없으면 생성
            if df.shape[1] < 4:
                print("Result column (Column D) not found. Adding it...")
                # 컬럼 이름이 중복되지 않게 'Generated SQL' 또는 인덱스로 추가
                df['Generated SQL'] = ""

            # Validation: Ensure data exists up to row 70 (index 68 in 0-based data rows)
            row_70_val = df.iloc[68, 2] if len(df) >= 69 else "N/A"
            print(f"Check Row 70, Column C: {row_70_val}")
            
            if len(df) < 69 or pd.isna(df.iloc[68, 2]):
                print(f"Error: Column C does not have data up to row 70. Terminating program.")
                sys.exit(1)

            # C열은 2번 인덱스, D열은 3번 인덱스
            # 2행부터 70행까지 (0-indexed: 1부터 69까지)
            start_row = 1
            end_row = 70
            
            for i in range(start_row, end_row):
                if i >= len(df):
                    break
                    
                query = df.iloc[i, 2] # C열 (질의)
                if pd.isna(query) or not str(query).strip():
                    print(f"Row {i+1}: Empty query, skipping.")
                    continue
                
                print(f"\n[{i+1}/{end_row}] Processing: {query}")
                
                try:
                    # 파이프라인 실행
                    generated_sql = run_pipeline(str(query))
                    
                    # 결과 D열에 저장
                    df.iloc[i, 3] = generated_sql
                    print(f"Row {i+1}: Success.")
                    
                except Exception as e:
                    print(f"Row {i+1}: Failed - {e}")
                    df.iloc[i, 3] = f"Error: {e}"
                    
                # 매 10개 행마다 중간 저장 (안전장치)
                if (i + 1) % 10 == 0:
                    df.to_excel(excel_path, sheet_name=sheet_name, index=False)
                    print(f"--- Intermediate save at row {i+1} ---")

            # 최종 저장
            print(f"\nFinal saving to {excel_path}...")
            df.to_excel(excel_path, sheet_name=sheet_name, index=False)
            print("All tasks completed successfully.")
            
        except Exception as e:
            print(f"Critical Excel Error: {e}")
