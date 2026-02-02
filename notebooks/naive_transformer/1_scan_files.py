import pandas as pd
import os

def peek_data(directory):
    # 关键文件名列表
    target_files = [
        'train.csv', 
        'test.csv', 
        'Sentences_Oare_FirstWord_LinNum.csv', 
        'OA_Lexicon_eBL.csv', 
        'published_texts.csv'
    ]
    
    print("="*50)
    print("DATASET PEEK REPORT")
    print("="*50)
    
    for filename in target_files:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            print(f"\n[FILE]: {filename}")
            try:
                # 读取前3行
                df = pd.read_csv(filepath, nrows=3)
                print(f"Columns: {list(df.columns)}")
                print("-" * 30)
                # 打印前3行数据
                print(df.to_string(index=False))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
            print("-" * 50)
        else:
            print(f"\n[MISSING]: {filename} not found in {directory}")

# 运行检查（假设文件夹在当前目录下）
peek_data('deep-past-initiative-machine-translation')