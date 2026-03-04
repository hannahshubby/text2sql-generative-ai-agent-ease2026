import os
import json
import sys
from typing import Dict, Any
import importlib.util
from pathlib import Path

# Ensure we can import config and other scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import CFG, TERM_LEXICON_JSON, TTL_CODE_INDEX_JSON, HIERARCHY_JSON, JOIN_GRAPH_JSON, COL_TO_TABLES_JSON, OUT_DIR
from common_io import load_json, dump_json

def import_module_from_path(module_name: str, file_name: str):
    file_path = os.path.join(current_dir, file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import agents
agent1 = import_module_from_path("agent1", "1_llm_understand_query.py")
agent2 = import_module_from_path("agent2", "2_ground_from_llm_understanding.py")
agent3 = import_module_from_path("agent3", "3_final_select.py")
agent4 = import_module_from_path("agent4", "4_confirm_finalize.py")
agent5_linking = import_module_from_path("agent5_linking", "5_table_linking_engine.py")
agent5_verifier = import_module_from_path("agent5_verifier", "5_5_join_semantic_verifier.py")
agent6 = import_module_from_path("agent6", "6_sql_planner_agent.py")
agent7 = import_module_from_path("agent7", "7_sql_synthesizer_agent.py")
# agent9 = import_module_from_path("agent9", "9_sql_synthesizer_agent_v2.py") # This one requires interim files

def run_pipeline(user_query: str) -> str:
    print(f"\n--- [Step 1] LLM Understand Query ---")
    CFG.sample_query = user_query
    try:
        out1 = agent1.main(CFG)
    except Exception as e:
        print(f" [!] Step 1 Failed: {e}")
        out1 = {"user_query": user_query, "parsed": {}}
    
    # Ensure 'user_query' exists
    if "user_query" not in out1:
        out1["user_query"] = user_query
        
    # Load static data for Step 2
    term_lex = load_json(TERM_LEXICON_JSON)
    ttl_idx = load_json(TTL_CODE_INDEX_JSON)
    
    print(f"--- [Step 2] Grounding from LLM Understanding ---")
    try:
        out2 = agent2.main(CFG, term_lex, ttl_idx, out1)
    except Exception as e:
        print(f" [!] Step 2 Failed: {e}")
        out2 = {"inputQuery": user_query, "termMentions": [], "codeMentions": []}
    
    print(f"--- [Step 3] Final Selection ---")
    try:
        out3 = agent3.main(CFG, out2)
    except Exception as e:
        print(f" [!] Step 3 Failed: {e}")
        out3 = out2 # Fallback to original
    
    print(f"--- [Step 4] Confirm & Finalize ---")
    try:
        out4_full, out4_min = agent4.main(CFG, out3, ttl_idx, term_lex)
    except Exception as e:
        print(f" [!] Step 4 Failed: {e}")
        out4_min = {"confirmed": {"terms": [], "codes": []}}
    
    # Load static data for Step 5
    hierarchy = load_json(HIERARCHY_JSON)
    join_graph = load_json(JOIN_GRAPH_JSON)
    col_to_tabs = load_json(COL_TO_TABLES_JSON)
    
    print(f"--- [Step 5] Table Linking ---")
    try:
        out5 = agent5_linking.main(out4_min, col_to_tabs, join_graph, hierarchy)
        if "error" in out5 or "joins" not in out5:
             raise ValueError(out5.get("error", "Unknown linking error"))
    except Exception as e:
        print(f" [!] Step 5 Failed ({e}). Using DUMMY table structure.")
        out5 = {
            "root_anchor": "UNKNOWN_TABLE",
            "selected_tables": [{"table": "UNKNOWN_TABLE", "tier": 1}],
            "joins": [],
            "column_bindings": {},
            "meta": {"fallback": True}
        }
    
    print(f"--- [Step 5.5] Join Semantic Verifier ---")
    try:
        step1_for_verifier = {"user_query": user_query}
        out5_verified = agent5_verifier.main(CFG, step1_for_verifier, out5)
    except Exception as e:
        print(f" [!] Step 5.5 Failed: {e}")
        out5_verified = out5 # Fallback to unverified joins
    
    print(f"--- [Step 6] SQL Planner ---")
    try:
        out6 = agent6.main(CFG, out1, out4_min, out5_verified)
    except Exception as e:
        print(f" [!] Step 6 Failed: {e}")
        out6 = {"where_sections": [], "order_by_section": []}
    
    print(f"--- [Step 7] SQL Synthesizer (LLM) ---")
    try:
        out7 = agent7.main(CFG, out1, out5_verified, out6)
    except Exception as e:
        print(f" [!] Step 7 Failed: {e}")
        out7 = {"sql": "-- SQL Synthesis failed internally"}
    
    final_sql = out7.get("sql", "Error: SQL not generated")
    print(f"\n[Generated SQL (Step 7)]\n{final_sql}")
    
    # Save results
    dump_json(out7, OUT_DIR / "final_sql.json")
    
    return final_sql

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Error: 'pandas'와 'openpyxl' 라이브러리가 필요합니다. 'pip install pandas openpyxl'을 실행해주세요.")
        sys.exit(1)

    print("\n" + "="*50)
    print("   SQL Generative AI Agent (A3 Pipeline)")
    print("="*50)
    print(" 1. Manual Query (수기 질의 입력)")
    print(" 2. Excel Batch Process (엑셀 배치 작업)")
    print("="*50)
    
    choice = input("선택하세요 (1 또는 2, 기본값 1): ").strip() or "1"

    if choice == "1":
        print("\n[수기 질의 모드] 종료하려면 'exit'를 입력하세요.")
        while True:
            query = input("\n질의를 입력하세요: ").strip()
            if query.lower() == 'exit':
                print("프로그램을 종료합니다.")
                break
            if not query:
                continue
            
            print(f"\n--- Processing: {query} ---")
            try:
                run_pipeline(query)
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                
    elif choice == "2":
        # Excel Batch Process
        excel_path = Path(current_dir) / "fewShotSample_260226.xlsx"
        sheet_name = "FewShotSample"
        
        try:
            print(f"Loading Excel from {excel_path}...")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
            if df.shape[1] < 4:
                print("Result column (Column D) not found. Adding it...")
                df['Generated SQL'] = ""
            
            for i in range(len(df)):
                query = df.iloc[i, 2] # C column
                if pd.isna(query) or not str(query).strip():
                    continue
                
                print(f"\n[{i+1}/{len(df)}] Processing: {query}")
                
                try:
                    generated_sql = run_pipeline(str(query))
                    df.iloc[i, 3] = generated_sql # D column
                except Exception as e:
                    print(f"Row {i+1} failed: {e}")
                    df.iloc[i, 3] = f"Error: {e}"
                
                if (i + 1) % 5 == 0:
                    df.to_excel(excel_path, sheet_name=sheet_name, index=False)
                    print(f"--- Intermediate save at row {i+1} ---")

            # Final save
            print(f"\nFinal saving to {excel_path}...")
            df.to_excel(excel_path, sheet_name=sheet_name, index=False)
            print("Batch processing completed.")
            
        except Exception as e:
            print(f"Critical Excel Error: {e}")
            traceback.print_exc()
    else:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")
