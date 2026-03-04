import os
import json
import pandas as pd
import sys
from pathlib import Path

# Set base directory
BASE_DIR = Path(r"d:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A2\UserAgent")
sys.path.append(str(BASE_DIR))

# Import agents
from agents_wrapper import run_pipeline

def main():
    excel_path = BASE_DIR / "fewShotSample_260226.xlsx"
    sheet_name = "FewShotSample"
    
    print(f"Loading Excel from {excel_path}...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Process rows 2 to 70 (0-indexed: 1 to 69)
    start_row = 1
    end_row = 70 # exclusive in my loop below if I use range(1, 70)
    
    for i in range(start_row, end_row):
        query = df.iloc[i, 2] # Column C is index 2
        if pd.isna(query) or not str(query).strip():
            print(f"Row {i+1}: Empty query, skipping.")
            continue
            
        print(f"\n[{i+1}/70] Processing query: {query}")
        
        try:
            sql_result = run_pipeline(str(query))
            df.iloc[i, 3] = sql_result # Column D is index 3
            print(f"Row {i+1}: Success!")
        except Exception as e:
            print(f"Row {i+1}: Failed with error: {e}")
            df.iloc[i, 3] = f"Error: {str(e)}"
            
    print(f"\nSaving results to {excel_path}...")
    df.to_excel(excel_path, sheet_name=sheet_name, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
