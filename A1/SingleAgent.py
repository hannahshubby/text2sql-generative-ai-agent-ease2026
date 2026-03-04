import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import time
import pandas as pd
from langchain_core.prompts import PromptTemplate

# BGE-m3
model_name = 'BAAI/bge-m3'
model = SentenceTransformer(model_name)

print(f"모델 로드 완료: {model_name}")
print(f"임베딩 차원: {model.get_sentence_embedding_dimension()}")

def search(query: str, index: faiss.Index, chunks: List[Dict], model: SentenceTransformer, top_k: int = 5, chunk_type_filter: str = None) -> List[Dict]:
    """
    쿼리로 검색하여 상위 k개 결과 반환
    chunk_type_filter: 특정 타입만 필터링 (예: 'table_schema_enriched', 'code', 'term')
    """
    # 쿼리 임베딩
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)

    # 필터링이 있으면 더 많이 검색해서 필터 후 top_k 추출
    search_k = top_k * 200 if chunk_type_filter else top_k
    scores, indices = index.search(query_embedding, search_k)

    # 결과 정리
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx]

            # 타입 필터링
            if chunk_type_filter and chunk.get('chunk_type') != chunk_type_filter:
                continue

            result = {
                'rank': len(results) + 1,
                'score': float(score),
                'chunk_id': chunk.get('chunk_id', ''),
                'chunk_type': chunk.get('chunk_type', ''),
                'text': chunk['text'][:500] + '...' if len(chunk['text']) > 500 else chunk['text']
            }
            results.append(result)

            if len(results) >= top_k:
                break

    return results

def search_baseline1(query: str, top_k: int = 5) -> List[Dict]:
    """
    Baseline-1 (스키마만) 검색
    """
    # 쿼리 임베딩
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # 검색
    scores, indices = index_b1.search(query_embedding, top_k)
    
    # 결과 정리
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(baseline1_chunks):
            chunk = baseline1_chunks[idx]
            results.append({
                'rank': len(results) + 1,
                'score': float(score),
                'chunk_id': chunk.get('chunk_id', ''),
                'chunk_type': chunk.get('chunk_type', ''),
                'text': chunk['text'][:500] + '...' if len(chunk['text']) > 500 else chunk['text']
            })
    
    return results


def search_baseline2(query: str, model: SentenceTransformer, top_k_each: int = 3) -> Dict:
    """
    SQL 생성을 위한 다중 검색 (테이블 + 코드 + 용어 병합)
    """
    # 테이블 검색
    tables = search(query, index_b2, baseline2_chunks, model,
                    top_k=top_k_each, chunk_type_filter='table_schema_enriched')

    # 코드 검색
    codes = search(query, index_b2, baseline2_chunks, model,
                   top_k=top_k_each, chunk_type_filter='code')

    # 용어 검색 (물리명 힌트용)
    terms = search(query, index_b2, baseline2_chunks, model,
                   top_k=top_k_each, chunk_type_filter='term')

    return {
        'tables': tables,
        'codes': codes,
        'terms': terms
    }



def func_search_baseline2(query: str, top_k_each: int = 3):
    """
    SQL 에이전트에게 전달할 컨텍스트 출력
    """
    results = search_baseline2(query, model, top_k_each)
    
    return results


def print_baseline2(query: str, top_k_each: int = 3):
    """
    SQL 에이전트에게 전달할 컨텍스트 출력
    """
    print("=" * 70)
    print(f"쿼리: {query}")
    print("=" * 70)

    results = search_baseline2(query, model, top_k_each)

    print("\n[관련 테이블]")
    print("-" * 50)
    for r in results['tables']:
        print(f"#{r['rank']} (score: {r['score']:.4f}) {r['chunk_id']}")
        print(f"{r['text'][:300]}...\n")

    print("\n[관련 코드]")
    print("-" * 50)
    for r in results['codes']:
        print(f"#{r['rank']} (score: {r['score']:.4f}) {r['chunk_id']}")
        print(f"{r['text'][:300]}...\n")

    print("\n[관련 용어 (물리명 힌트)]")
    print("-" * 50)
    for r in results['terms']:
        print(f"#{r['rank']} (score: {r['score']:.4f}) {r['chunk_id']}")
        print(f"{r['text']}\n")

    return results

# 청크 데이터 로드
with open(r'd:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A1\baseline1_schema_only.json', 'r', encoding='utf-8') as f:
    baseline1_chunks = json.load(f)

with open(r'd:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A1\baseline2_with_dictionary.json', 'r', encoding='utf-8') as f:
    baseline2_chunks = json.load(f)

print(f"Baseline-1 청크 수: {len(baseline1_chunks)}")
print(f"Baseline-2 청크 수: {len(baseline2_chunks)}")

index_b1 = faiss.read_index(r'd:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A1\baseline1_faiss.index')
index_b2 = faiss.read_index(r'd:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A1\baseline2_faiss.index')

# 테스트 쿼리 1(baseline2)
func_search_baseline2("사용 중인 금융기관 중 금융기관코드가 '088'이고 [contTrkey] 이상인 금융기관의 코드와 명칭을 [thQryC]건만 조회해줘")

#search_result=func_search_baseline2("화면ID가 :scrId인 화면에 대해 권한 사용자, 통제부서코드, 통제버튼구분코드, 통제버튼권한여부와 함께 작업자ID/작업단말ID/작업상세일시/GUID를 조회해줘")
#print(search_result)

#from langchain.prompts import PromptTemplate

SQL_PROMPT = PromptTemplate(
    input_variables=["schema_context", "question"],
    template="""
You are a senior PostgreSQL database engineer.

Using ONLY the schema information below, write a correct SQL query.

CRITICAL RULES (VERY IMPORTANT):
- NEVER translate, rename, or paraphrase column names or table names
- Column names and aliases MUST match exactly what is defined in the schema
- Do NOT create business-friendly or human-readable aliases
- If an alias is required, use the original column name or a simple technical alias
  (e.g., tablename.columnname)

JOIN RULES:
- If the query involves JOINs:
  - EVERY column in SELECT, WHERE, GROUP BY, ORDER BY
    MUST be fully qualified using table alias (alias.column_name)

GENERAL RULES:
- Return ONLY the SQL query
- Do NOT explain
- Do NOT use markdown
- Do NOT invent tables or columns
- Use PostgreSQL syntax only

Schema:
{schema_context}

Question:
{question}

SQL:
"""
)

from langchain_openai import ChatOpenAI

OPENAI_API_KEY = ""


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
)

def generate_sql(question: str, schema_context: str) -> str:
    prompt = SQL_PROMPT.format(
        schema_context=schema_context,
        question=question
    )
    response = llm.invoke(prompt)
    return response.content.strip()

# Excel 파일 로드 및 질의 처리
excel_file = r'd:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A1\fewShotSample_260226.xlsx'
sheet_name = 'FewShotSample'

print(f"\nExcel 파일 로드 중: {excel_file}")
# 모든 시트 로드 (보존을 위해)
all_sheets = pd.read_excel(excel_file, sheet_name=None)
df = all_sheets[sheet_name]

# C열(index 2)에서 질문 가져오기, D열(index 3)에 결과 저장
generated_sqls = []

# Loop through all rows in the sheet
for idx, row in df.iterrows():
    # C열은 index 2
    question = str(row.iloc[2]) if not pd.isna(row.iloc[2]) else ""
    
    if not question.strip() or question == "nan":
        generated_sqls.append("")
        continue
        
    print(f"[{idx+1}/{len(df)}] 질의 처리 중: {question[:50]}...")
    
    # 1. context 검색
    search_result = func_search_baseline2(question)
    
    # 2. SQL 생성
    try:
        sql_result = generate_sql(
            question=question,
            schema_context=search_result
        )
    except Exception as e:
        print(f"Error generating SQL for index {idx}: {e}")
        sql_result = f"Error: {e}"
    
    generated_sqls.append(sql_result)
    print(f"Generated SQL: {sql_result[:100]}...")

# D열(index 3) 데이터 업데이트
# 데이터프레임의 컬럼 수가 부족할 수 있으므로 보정
if df.shape[1] <= 3:
    # 새로운 컬럼 추가
    col_name = "Generated SQL"
    df[col_name] = generated_sqls
else:
    # 기존 D열(index 3) 위치에 덮어쓰기
    df.iloc[:, 3] = generated_sqls

# 업데이트된 데이터프레임을 시트 목록에 반영
all_sheets[sheet_name] = df

# 파일 저장 (다른 시트들 포함)
print(f"Excel 파일 저장 중...")
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    for s_name, s_df in all_sheets.items():
        s_df.to_excel(writer, sheet_name=s_name, index=False)

print("모든 작업이 완료되었습니다.")

