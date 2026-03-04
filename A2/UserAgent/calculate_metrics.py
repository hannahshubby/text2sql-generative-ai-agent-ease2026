import pandas as pd
import json
import os
import time
from typing import Dict, Any
from openai import OpenAI

# 파일 경로 및 설정
EXCEL_FILE = r'D:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A2\UserAgent\fewShotSample_260226_A2.xlsx'

# OpenAI 설정
API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o"

def evaluate_sql_with_llm(gold_sql: str, pred_sql: str) -> Dict[str, Any]:
    """LLM을 사용하여 두 SQL의 논리적 일치도를 평가합니다."""
    
    if not pred_sql or pd.isna(pred_sql) or str(pred_sql).lower().startswith('error'):
        return {
            "TLA": [1, 0, 0, 0, 0, 0], "CLA": [1, 0, 0, 0, 0, 0], "LMA": [1, 0, 0, 0, 0, 0], 
            "Complexity": 0, "Rationale": "Prediction failed"
        }

    system_prompt = (
        "You are a SQL auditor. Compare a 'Gold SQL' with a 'Predicted SQL' and evaluate their structural match.\n"
        "Provide your evaluation in strict JSON format with the following keys:\n"
        "- TLA: [Gold_Table_Cnt, Pred_Table_Cnt, Match_Table_Cnt, Precision, Recall, F1]\n"
        "- CLA: [Gold_Col_Cnt, Pred_Col_Cnt, Match_Col_Cnt, Precision, Recall, F1]\n"
        "- LMA: [Gold_Logic_Cnt, Pred_Logic_Cnt, Match_Logic_Cnt, Precision, Recall, F1]\n"
        "- Complexity: Total count of columns, tables, and conditions in Predicted SQL.\n"
        "- Rationale: A brief explanation of errors or differences.\n\n"
        "Rules:\n"
        "1. Table/Column matching is based on semantic usage in the query.\n"
        "2. Logic (LMA) includes WHERE conditions, GROUP BY columns, and Aggregate functions.\n"
        "3. Metrics must be floats between 0.0 and 1.0."
    )

    user_content = f"### Gold SQL:\n{gold_sql}\n\n### Predicted SQL:\n{pred_sql}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {
            "TLA": [0, 0, 0, 0, 0, 0], "CLA": [0, 0, 0, 0, 0, 0], "LMA": [0, 0, 0, 0, 0, 0], 
            "Complexity": 0, "Rationale": f"LLM error: {e}"
        }

def get_column_letter(n):
    """숫자 인덱스를 엑셀 열 문자(A, B, C...)로 변환"""
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string

def main():
    print("Excel 데이터 로드 중...")
    all_sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)
    df = all_sheets['FewShotSample'].copy()
    
    detail_results = []
    
    print(f"총 {len(df)}건의 SQL을 LLM으로 평가 중...")
    for idx, row in df.iterrows():
        gold_sql = str(row.iloc[1]) # B열
        pred_sql = str(row.iloc[3]) # D열
        
        print(f"[{idx+1}/{len(df)}] Evaluating row...")
        res = evaluate_sql_with_llm(gold_sql, pred_sql)
        
        tla = res.get("TLA", [0]*6)
        cla = res.get("CLA", [0]*6)
        lma = res.get("LMA", [0]*6)
        
        detail_results.append({
            # Table Metrics (TLA)
            'TLA_Gold_Cnt': tla[0], 'TLA_Pred_Cnt': tla[1], 'TLA_Match_Cnt': tla[2], 
            'TLA_Precision': tla[3], 'TLA_Recall': tla[4], 'TLA_F1': tla[5],
            
            # Column Metrics (CLA)
            'CLA_Gold_Cnt': cla[0], 'CLA_Pred_Cnt': cla[1], 'CLA_Match_Cnt': cla[2],
            'CLA_Precision': cla[3], 'CLA_Recall': cla[4], 'CLA_F1': cla[5],
            
            # Logic Metrics (LMA)
            'LMA_Gold_Cnt': lma[0], 'LMA_Pred_Cnt': lma[1], 'LMA_Match_Cnt': lma[2],
            'LMA_Precision': lma[3], 'LMA_Recall': lma[4], 'LMA_F1': lma[5],
            
            'SQL_Complexity': res.get("Complexity", 0),
            'LLM_Rationale': res.get("Rationale", "")
        })
        
        # API 과금 및 속도 조절 (필요시)
        # time.sleep(0.1)
        
    df_detail = pd.concat([df.reset_index(drop=True), pd.DataFrame(detail_results)], axis=1)
    
    start_row = 2
    end_row = len(df_detail) + 1
    orig_cols = df.shape[1] 
    
    # 위치 계산 (TLA 6개 + CLA 6개 + LMA 6개 + Complexity 1개 + Rationale 1개)
    tla_f1_col = get_column_letter(orig_cols + 6)
    cla_f1_col = get_column_letter(orig_cols + 12)
    lma_f1_col = get_column_letter(orig_cols + 18)
    comp_col = get_column_letter(orig_cols + 19)
    
    summary_data = {
        'Indicator': [
            'Total Samples',
            'Column Linking Accuracy (CLA) Avg',
            'Table Linking Accuracy (TLA) Avg',
            'Logic Matching Accuracy (LMA) Avg',
            'Average SQL_Complexity (Conciseness)'
        ],
        'Result': [
            f"=COUNTA(Result_Detail!A{start_row}:A{end_row})",
            f"=AVERAGE(Result_Detail!{cla_f1_col}{start_row}:{cla_f1_col}{end_row})",
            f"=AVERAGE(Result_Detail!{tla_f1_col}{start_row}:{tla_f1_col}{end_row})",
            f"=AVERAGE(Result_Detail!{lma_f1_col}{start_row}:{lma_f1_col}{end_row})",
            f"=AVERAGE(Result_Detail!{comp_col}{start_row}:{comp_col}{end_row})"
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    
    print("Excel 저장 중...")
    with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_detail.to_excel(writer, sheet_name='Result_Detail', index=False)
        df_summary.to_excel(writer, sheet_name='Result_Summary', index=False)
        
    print(f"작업 완료! LLM 기반 SQL 평가 지표가 생성되었습니다.")

if __name__ == "__main__":
    main()
