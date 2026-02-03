import pandas as pd
import os
import re
import glob

# ================= 配置区 =================
# 自动定位到 deep-past 数据集目录
CURRENT_DIR = os.getcwd()
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "data", "deep-past-initiative-machine-translation")

# ================= 核心分析引擎 =================

def analyze_text_fingerprint(series, col_name):
    """
    深度文本法医分析：分析一列文本的符号特征
    """
    # 取前 2000 行非空数据作为样本，兼顾速度与代表性
    valid_data = series.dropna().astype(str)
    if valid_data.empty:
        return "    - [空列]"
    
    sample_text = " ".join(valid_data.head(2000))
    
    # 1. 括号风格 (关键：决定是否需要做 () -> {} 映射)
    round_brackets = len(re.findall(r'\(.*?\)', sample_text))
    curly_brackets = len(re.findall(r'\{.*?\}', sample_text))
    
    # 2. 变音符号 (关键：决定是否需要保护 š, ṣ, ḫ)
    h_chars = len(re.findall(r'[ḫḪ]', sample_text))
    s_chars = len(re.findall(r'[šŠ]', sample_text))
    
    # 3. 下标 (关键：决定是否需要 Unicode 归一化)
    uni_subs = len(re.findall(r'[₀-₉]', sample_text))
    
    # 4. 缺损标记
    gaps = sample_text.count('[x]') + sample_text.count('...')
    
    report = f"    [🔬 列分析: {col_name}]\n"
    report += f"      - 确定值风格 : 圆括号={round_brackets} vs 花括号={curly_brackets}"
    
    if round_brackets > 0 and curly_brackets == 0:
        report += " ⚠️ (需启用 V11 转换)"
    elif curly_brackets > 0:
        report += " ✅ (标准格式)"
        
    report += f"\n      - 文明指纹   : ḫ/Ḫ={h_chars}, š/Š={s_chars}, 下标={uni_subs}"
    report += f"\n      - 缺损标记   : {gaps} 处"
    
    return report

def scan_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\n>>> 正在扫描文件: {filename}")
    
    try:
        # 强制 UTF-8 读取
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"    - 形状: {df.shape}")
        print(f"    - 列名: {df.columns.tolist()}")
        
        # 智能探测：寻找包含 'translit', 'transla', 'text', 'spelling' 的列进行深度分析
        target_cols = [c for c in df.columns if any(x in c.lower() for x in ['translit', 'transla', 'text', 'spelling', 'form'])]
        
        if target_cols:
            print(f"    - 命中核心文本列: {target_cols}")
            for col in target_cols:
                print(analyze_text_fingerprint(df[col], col))
        else:
            print("    - (未检测到明显的阿卡德语/英语文本列，跳过深度分析)")
            
        # 价值评估
        cols_lower = [c.lower() for c in df.columns]
        has_source = any('translit' in c for c in cols_lower)
        has_target = any('transla' in c or 'eng' in c for c in cols_lower)
        
        if has_source and has_target:
            print(f"    🌟 [高价值] 发现潜在的平行语料 (Source + Target)!")
        elif has_source:
            print(f"    🔶 [中价值] 仅发现转写文本 (可用作预训练/单语数据)")
            
        return df  # 返回 DataFrame 用于后续关联分析

    except Exception as e:
        print(f"    ❌ 读取失败: {e}")
        return None

def check_relationships(data_map):
    print(f"\n{'='*20} 🔗 文件关联性图谱分析 {'='*20}")
    
    # 1. 核心关联: Train <-> Sentences
    if 'train.csv' in data_map and 'Sentences_Oare_FirstWord_LinNum.csv' in data_map:
        train = data_map['train.csv']
        sent = data_map['Sentences_Oare_FirstWord_LinNum.csv']
        
        # 检查 train.oare_id 和 sent.text_uuid
        common = set(train['oare_id']).intersection(set(sent['text_uuid']))
        coverage = len(common) / len(train) * 100
        print(f"  [Train <-> Sentences]")
        print(f"    - 关联键: train['oare_id'] == sent['text_uuid']")
        print(f"    - 匹配 ID 数: {len(common)} (覆盖率: {coverage:.2f}%)")
        if coverage < 10:
            print("    ⚠️ 警告: 覆盖率极低，说明大部分训练集文档没有对应的句子切分数据！")
    
    # 2. 潜在关联: Published Texts (如果有)
    if 'published_texts.csv' in data_map:
        pub = data_map['published_texts.csv']
        print(f"  [Published Texts 概况]")
        print(f"    - 总行数: {len(pub)}")
        # 看看有没有 ID 可以跟其他表连
        if 'id' in pub.columns or 'uuid' in pub.columns:
            print(f"    - 可能的主键: {[c for c in pub.columns if 'id' in c.lower()]}")

# ================= 主程序 =================

def main():
    print(f"{'='*30} 全域数据资产法医级扫描 {'='*30}")
    print(f"目标目录: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 致命错误: 目录不存在！")
        return

    # 获取所有 csv 文件
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"发现 {len(csv_files)} 个 CSV 文件待审计。\n")
    
    data_map = {}
    
    # 逐个扫描
    for fpath in csv_files:
        filename = os.path.basename(fpath)
        df = scan_file(fpath)
        if df is not None:
            data_map[filename] = df
            
    # 关联分析
    check_relationships(data_map)
    
    print(f"\n{'='*30} 扫描结束 {'='*30}")
    print(">>> 核心行动指南:")
    print("1. 如果 Train/Test 的括号风格是 '圆括号'，预处理必须包含 replace('(', '{')。")
    print("2. 如果 Sentences 的覆盖率低，需要检查是否可以通过 published_texts.csv 补充数据。")
    print("3. 如果发现 '中价值' 文件，考虑将其加入预训练语料库。")

if __name__ == "__main__":
    main()