from __future__ import annotations

import copy
import pandas as pd
import re, os, json, pprint
from openai import OpenAI
from getpass import getpass

from dataclasses import dataclass
from typing import List, Literal, Any, Optional, Dict, Tuple, Protocol
from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator
from datetime import datetime

from pathlib import Path



from rapidfuzz import fuzz, process
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from abc import ABC, abstractmethod


from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, SKOS

from collections import defaultdict, deque




########
import sqlglot
from sqlglot import exp
########



from logging_config import setup_logging
import logging


setup_logging()
logger = logging.getLogger(__name__)

# cache file names (build_term_index.py와 동일)
TERMS_JSONL = "terms.jsonl"
FAISS_INDEX = "faiss.index"
MANIFEST_JSON = "manifest.json"






##################################### 1. get term data to dataset ###################################################
def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="cp949", encoding_errors="ignore")

def get_term(csv_path=None):
    # 1) CSV 로드
    if csv_path is None:
        try:
            from config import TERM_COLLECT_CSV
            csv_path = str(TERM_COLLECT_CSV)
        except ImportError:
            csv_path = "csv/term_collect.csv"
            
    df = read_csv_safely(csv_path)

    # 2) 컬럼명 매핑 (당신 CSV 컬럼명 기준)
    df = df.rename(columns={
        "용어명": "term_name_ko",
        "물리명": "physical_column_name",
        "인포타입": "info_type",
        "코드명": "code_name",
        "용어정의": "definition",
    })

    df = df[["term_name_ko","physical_column_name","info_type","code_name","definition"]]
    return df



######################################### 2. query canonicalization ###################################################

@dataclass
class RewriteResult:
    developer_rewrite: str
    clarify: str


@dataclass
class ClarificationTurn:
    question: str
    answer: str


class CanonicalRewriteAgent:
    CANONICAL_SYSTEM_PROMPT = """You are a Canonical Rewrite Agent for enterprise Text-to-SQL.
    Your job is to rewrite the user's request into a developer-ready, unambiguous question.

    STRICT RULES:
    - Preserve ALL constraints and requested outputs from the original query. Do not drop, weaken, or add constraints.
    - Make negation and exclusion explicit in Korean so a developer can implement it without guessing.
    - When the user uses parenthetical expressions, treat them as synonym/alias hints and reflect them without losing meaning.
    - Do NOT mention physical column names, table names, schema names, code names, or any implementation identifiers.
    - Use the provided clarification answers to eliminate ambiguity. If ambiguity remains, ask the next best clarifying question.
    - Output MUST follow this exact 2-line format:

    developer_rewrite: <one paragraph in Korean. Use structured numbering (1)(2)(3) if helpful.>
    clarify: <a short question in Korean> OR "없음"

    Return only these two lines. No extra text.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        api_key = ""
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def rewrite(self, user_query: str, clarifications: List[ClarificationTurn], *, temperature: float = 0.0) -> RewriteResult:
        # Build a single user message that contains the original query + Q/A history.
        # (This keeps the agent deterministic and easy to debug.)
        qa_block = ""
        if clarifications:
            qa_lines = []
            for i, t in enumerate(clarifications, start=1):
                qa_lines.append(f"Q{i}: {t.question}")
                qa_lines.append(f"A{i}: {t.answer}")
            qa_block = "\n\n[Clarification Q/A]\n" + "\n".join(qa_lines)

        user_content = f"[Original Query]\n{user_query}{qa_block}"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
        )
        text = (resp.choices[0].message.content or "").strip()

        dev, clar = self._parse_contract(text)
        self._guard_no_physical_artifacts(dev + "\n" + clar)

        return RewriteResult(developer_rewrite=dev, clarify=clar)

    @staticmethod
    def _parse_contract(text: str) -> Tuple[str, str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        dev_line = next((ln for ln in lines if ln.lower().startswith("developer_rewrite:")), None)
        clar_line = next((ln for ln in lines if ln.lower().startswith("clarify:")), None)
        if not dev_line or not clar_line:
            raise RuntimeError(f"Model did not follow the required 2-line format.\n\nRaw:\n{text}")

        dev = dev_line.split(":", 1)[1].strip()
        clar = clar_line.split(":", 1)[1].strip() or "없음"
        if not dev:
            raise RuntimeError(f"Empty developer_rewrite.\n\nRaw:\n{text}")
        return dev, clar

    @staticmethod
    def _guard_no_physical_artifacts(text: str) -> None:
        if re.search(r"\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b", text):
            raise RuntimeError("Output contains code-like identifiers (UPPER_SNAKE_CASE).")
        if re.search(r"\b[A-Za-z]+\.[A-Za-z_]+\b", text):
            raise RuntimeError("Output contains table.column-like identifiers.")


class ClarifyLoop:
    """
    Controls the iterative clarification loop.
    - Call resolve() to get the next question or the final rewrite.
    - Feed user answers via submit_answer().
    """
    def __init__(self, agent: CanonicalRewriteAgent, user_query: str, max_rounds: int = 5):
        self.agent = agent
        self.user_query = user_query
        self.max_rounds = max_rounds
        self.turns: List[ClarificationTurn] = []
        self.last_result: Optional[RewriteResult] = None

    def step(self) -> RewriteResult:
        if len(self.turns) >= self.max_rounds:
            # Stop asking forever; return best-effort rewrite.
            return self.last_result or self.agent.rewrite(self.user_query, self.turns)

        self.last_result = self.agent.rewrite(self.user_query, self.turns)
        return self.last_result

    def needs_clarification(self) -> bool:
        if not self.last_result:
            return True
        return self.last_result.clarify.strip() != "없음"

    def get_question(self) -> str:
        if not self.last_result:
            raise RuntimeError("Call step() first.")
        return self.last_result.clarify

    def submit_answer(self, answer: str) -> None:
        if not self.last_result:
            raise RuntimeError("Call step() first.")
        q = self.last_result.clarify.strip()
        if q == "없음":
            return
        self.turns.append(ClarificationTurn(question=q, answer=answer))

    def final_rewrite(self) -> str:
        if not self.last_result:
            raise RuntimeError("Call step() first.")
        return self.last_result.developer_rewrite

def clarify_loop_colab_input(agent, user_query: str, max_rounds: int = 5):
    turns = []
    result = None

    for i in range(1, max_rounds + 1):
        result = agent.rewrite(user_query, turns, temperature=0.0)
        print(f"\n[Round {i}]")
        print("developer_rewrite:", result.developer_rewrite)
        print("clarify:", result.clarify)

        if result.clarify.strip() == "없음":
            break

        answer = input("답변을 입력하세요(끝내려면 stop):\n> ").strip()
        if answer.lower() == "stop":
            break

        turns.append(ClarificationTurn(question=result.clarify.strip(), answer=answer))

    return result.developer_rewrite if result else "", turns



def UserCanonicalRewriteAgent(UserQuery: str):
    agent = CanonicalRewriteAgent(model="gpt-4.1-mini") 
    query = UserQuery #"계좌 상태가 폐쇄, 이관(전출), 이관신청 상태가 아니고 실명 가명 구분코드가 기타가 아닌 계좌 중에, 사용 중인 계좌의 온라인 사용자 ID와 해당 사용자가 보유한 계좌번호를 조회해줘."


    final_rewrite, qa_log = clarify_loop_colab_input(agent, query, max_rounds=5)

    print("\n[FINAL developer_rewrite]")
    print(final_rewrite)

    print("\n[Q/A LOG]")
    for t in qa_log:
        print("-", t.question, "=>", t.answer)

    return final_rewrite


def extract_candidates_list(result: dict) -> list[str]:
    candidates = result.get("column_candidates", [])
    if candidates is None:
        return []
    if not isinstance(candidates, list):
        candidates = [candidates]
    return [str(x).strip() for x in candidates if str(x).strip()]


####################################### 3. column candidate extraction ###################################################





def extract_column_candidates(nl_query: str, model: str = "gpt-4.1-mini") -> dict:
    
    EXTRACT_COLUMN_CANDIDATES_SYSTEM_PROMPT = """
    You extract column candidates from a Korean NL request.
    Return JSON ONLY with a single key: "column_candidates".

    Rules:
    - Output must be valid JSON only. No markdown, no extra keys.
    - "column_candidates" must be a JSON array of strings.
    - Keep the original Korean/English surface form as it appears (normalise spaces minimally).
    - Deduplicate while preserving order.
    """
    
    api_key=""
    client = OpenAI(api_key=api_key)
    
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACT_COLUMN_CANDIDATES_SYSTEM_PROMPT},
            {"role": "user", "content": nl_query},
        ],
    )
    content = resp.choices[0].message.content
    data = json.loads(content)

    # Safety: enforce single key and list[str]
    cols = data.get("column_candidates", [])
    if not isinstance(cols, list):
        cols = []
    # force string + dedupe preserve order
    seen = set()
    cleaned = []
    for c in cols:
        if isinstance(c, str):
            cc = c.strip()
            if cc and cc not in seen:
                seen.add(cc)
                cleaned.append(cc)
    return {"column_candidates": cleaned}






####################################### 3. column candidate extraction ###################################################





def extract_column_candidates(nl_query: str, model: str = "gpt-4.1-mini") -> dict:
    
    EXTRACT_COLUMN_CANDIDATES_SYSTEM_PROMPT = """
    You extract column candidates from a Korean NL request.
    Return JSON ONLY with a single key: "column_candidates".

    Rules:
    - Output must be valid JSON only. No markdown, no extra keys.
    - "column_candidates" must be a JSON array of strings.
    - Keep the original Korean/English surface form as it appears (normalise spaces minimally).
    - Deduplicate while preserving order.
    """
    
    api_key=""
    client = OpenAI(api_key=api_key)
    
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACT_COLUMN_CANDIDATES_SYSTEM_PROMPT},
            {"role": "user", "content": nl_query},
        ],
    )
    content = resp.choices[0].message.content
    data = json.loads(content)

    # Safety: enforce single key and list[str]
    cols = data.get("column_candidates", [])
    if not isinstance(cols, list):
        cols = []
    # force string + dedupe preserve order
    seen = set()
    cleaned = []
    for c in cols:
        if isinstance(c, str):
            cc = c.strip()
            if cc and cc not in seen:
                seen.add(cc)
                cleaned.append(cc)
    return {"column_candidates": cleaned}






####################################   main   ####################################
#if __name__ == "__main__":
    

def main_extract_column_candidates(final_rewrite):
    #with open(r"interim_result\UserCanonicalRewrite.txt",
    #    "r", encoding="cp949") as f:
    #    final_rewrite = f.read().strip()


    print(f"=======> Loaded final_rewrite from aaa.txt: {final_rewrite}")

    #3) extract column candidates
    extract_result = extract_column_candidates(final_rewrite)
    print(json.dumps(extract_result, ensure_ascii=False, indent=2))
    column_candidates_list = extract_candidates_list(extract_result)
    print("=======> extract column candidates: ", column_candidates_list)
    return extract_result
    #output_path = Path(r"interim_result\2.column_candidates_list.txt")

    #output_path.write_text(
    #    "\n".join(column_candidates_list),
    #    encoding="utf-8"
    #)