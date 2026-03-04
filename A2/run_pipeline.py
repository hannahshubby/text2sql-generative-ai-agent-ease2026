from __future__ import annotations

import os
import json
import sys
import uuid
from typing import Any, Dict, Optional

from utils import load_db_storage, load_module_from_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

USER_AGENT_DIR = os.path.join(BASE_DIR, "UserAgent")


def _import(module_filename: str, module_name: str):
    return load_module_from_path(module_name, os.path.join(USER_AGENT_DIR, module_filename))


def _log(db, *, trace_id: str, run_id: int, step_id: str, agent_name: str, direction: str,
         payload: Dict[str, Any], status: str = "OK", error: Optional[str] = None, actor_type: str = "AI") -> None:
    db.save_agent_io(
        trace_id=trace_id,
        run_id=run_id,
        step_id=step_id,
        agent_name=agent_name,
        actor_type=actor_type,
        direction=direction,
        payload=payload,
        status=status,
        error_message=error,
    )


def run_all_steps(*, user_query: str, trace_id: Optional[str] = None) -> Dict[str, Any]:

    #trace_id = "4015d7bf-3551-48da-8154-394775071b60"
    #run_id = 4

    if trace_id is None:
        trace_id = str(uuid.uuid4())
        run_id = 0


    db = load_db_storage(BASE_DIR)

    print("trace_id="+trace_id)


    if run_id == 0:
          # Step1
          run_id += 1
          step1 = _import("1_UserCanonicalRewriteAgent.py", f"step1_{trace_id}")

          _log(db, trace_id=trace_id, run_id=run_id, step_id="1", agent_name="UserCanonicalRewriteAgent",
               direction="REQUEST", payload={"user_query": user_query})
          try:
               rewrite = step1.main_UserCanonicalRewriteAgent(user_query)
               _log(db, trace_id=trace_id, run_id=run_id, step_id="1", agent_name="UserCanonicalRewriteAgent",
                    direction="RESPONSE", payload={"developer_rewrite": rewrite})
          except Exception as e:
               _log(db, trace_id=trace_id, run_id=run_id, step_id="1", agent_name="UserCanonicalRewriteAgent",
                    direction="RESPONSE", payload={}, status="ERROR", error=str(e))
               raise




    if run_id == 1:
          # Step2
          run_id += 1
          step2 = _import("2_column_candidates_list.py", f"step2_{trace_id}")
          _log(db, trace_id=trace_id, run_id=run_id, step_id="2", agent_name="ColumnCandidatesListAgent",
               direction="REQUEST", payload={"developer_rewrite": rewrite})
          try:
               step2_extract_result = step2.extract_column_candidates(rewrite)
               #cols = step2.extract_candidates_list(extract_result)
               _log(db, trace_id=trace_id, run_id=run_id, step_id="2", agent_name="ColumnCandidatesListAgent",
                    direction="RESPONSE", payload=step2_extract_result)
          except Exception as e:
               _log(db, trace_id=trace_id, run_id=run_id, step_id="2", agent_name="ColumnCandidatesListAgent",
                    direction="RESPONSE", payload={}, status="ERROR", error=str(e))
               raise

    if run_id == 2:
          # Step3
          run_id += 1
          step3 = _import("3_column_term_candidates_list.py", f"step3_{trace_id}")
          _log(db, trace_id=trace_id, run_id=run_id, step_id="3", agent_name="ColumnTermCandidatesAgent",
               direction="REQUEST", payload=step2_extract_result)
          try:
               step3_bench = step3.main_column_term_candidates_list(step2_extract_result)
               _log(db, trace_id=trace_id, run_id=run_id, step_id="3", agent_name="ColumnTermCandidatesAgent",
                    direction="RESPONSE", payload=step3_bench)
          except Exception as e:
               _log(db, trace_id=trace_id, run_id=run_id, step_id="3", agent_name="ColumnTermCandidatesAgent",
                    direction="RESPONSE", payload={}, status="ERROR", error=str(e))
               raise

    if run_id == 3:
          # Step4
          run_id += 1
          #step4 = _import("4.OntologySearchSelection_semantic_checkpoints_specificity_policy.py", f"step4_{trace_id}")
          step4 = _import("4.OntologySearchSelection_semantic_checkpoints_specificity_policy_gates.py", f"step4_{trace_id}")

          
          ttl_path = os.path.join(BASE_DIR, "ttl", "financial_terms.ttl")
          print(ttl_path)
          _log(db, trace_id=trace_id, run_id=run_id, step_id="4", agent_name="OntologySearchSelectionAgent",
               direction="REQUEST", payload={"ttl_path": ttl_path})
          try:
               step4_sel = step4.main_ontology_search_selection(step3_bench, ttl_path)
               _log(db, trace_id=trace_id, run_id=run_id, step_id="4", agent_name="OntologySearchSelectionAgent",
                    direction="RESPONSE", payload=step4_sel)
          except Exception as e:
               _log(db, trace_id=trace_id, run_id=run_id, step_id="4", agent_name="OntologySearchSelectionAgent",
                    direction="RESPONSE", payload={}, status="ERROR", error=str(e))
               raise

    if run_id == 4:
          # Step5
          run_id += 1
          step5 = _import("5.llm_finalize_using_intent.py", f"step5_{trace_id}")
          _log(db, trace_id=trace_id, run_id=run_id, step_id="5", agent_name="LLMFinalizeAgent",
               direction="REQUEST", payload={"intent": rewrite, "in_data": step4_sel})
          try:
               step5_fin = step5.main_llm_finalize_using_intent(rewrite, step4_sel)
               _log(db, trace_id=trace_id, run_id=run_id, step_id="5", agent_name="LLMFinalizeAgent",
                    direction="RESPONSE", payload=step5_fin)
          except Exception as e:
               _log(db, trace_id=trace_id, run_id=run_id, step_id="5", agent_name="LLMFinalizeAgent",
                    direction="RESPONSE", payload={}, status="ERROR", error=str(e))
               raise


    #if run_id == 5:





    print ("DONE. trace_id=", trace_id)
    #return {"trace_id": trace_id, "final_rewrite": rewrite, "final_sql_json": final_sql}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py \"your query here\"")
        sys.exit(1)

    q = sys.argv[1]
    result = run_all_steps(user_query=q)
    #print("DONE. trace_id=", result["trace_id"])
    #print("Final SQL JSON:", json.dumps(result["final_sql_json"], ensure_ascii=False, indent=2))
