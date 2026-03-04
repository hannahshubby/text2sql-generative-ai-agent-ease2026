import pandas as pd
try:
    file_path = r'd:\GitHub\text2sql-generative-ai-agent-new\z.AB_TEST\A1\fewShotSample_260226.xlsx'
    df = pd.read_excel(file_path, sheet_name='FewShotSample')
    print("Columns:", df.columns.tolist())
    print("First 5 rows of Column C (index 2):")
    print(df.iloc[:5, 2])
except Exception as e:
    print(f"Error: {e}")
