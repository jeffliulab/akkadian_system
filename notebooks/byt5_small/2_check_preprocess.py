import pandas as pd
import re
import os
import sys

# ================= 配置区 =================
INPUT_FILE = "clean.csv"

# [规则集 1] 绝对禁止出现在 INPUT (阿卡德语) 的符号
# 包含点号(.)，因为官方文档说这是 word divider 要删
FORBIDDEN_INPUT_CHARS = ["!", "\\?", "/", ":", "\\.", "˹", "˺", "\\[", "\\]", "\\(", "\\)"]

# [规则集 2] 绝对禁止出现在 TARGET (英语) 的符号
# 不包含点号(.)和逗号(,)，因为英语句子需要标点
FORBIDDEN_TARGET_CHARS = ["!", "\\?", "/", ":", "˹", "˺", "\\[", "\\]", "\\(", "\\)"]

# [规则集 3] 必须强制映射的字符
FORBIDDEN_MAPPING = ["ḫ", "Ḫ"]

# [规则集 4] 必须保留的阿卡德语特征
AKKADIAN_CHARS = r"[šŠṣṢṭṬáéíúùÁÉÍÚÙ]"

# [规则集 5] 必须清洗掉的 URL 特征
URL_PATTERNS = r"http|https|www\.|search\?q=|aicuneiform"

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def audit_compliance(df):
    print_header("维度 1: 比赛规则合规性 (Compliance)")
    
    # [Check 1] 字符映射 (H/h Rule)
    mask_h = df['input_text'].str.contains(r'[ḫḪ]', regex=True)
    fail_h = mask_h.sum()
    print(f"1. 字符映射检查 (ḫ -> h):")
    if fail_h == 0:
        print(f"   ✅ PASS。无残留 'ḫ/Ḫ'。")
    else:
        print(f"   ❌ FAIL! 发现 {fail_h} 行残留。")

    # [Check 2] Input 端考古垃圾清理 (Strict Mode)
    # 移除合法的 <gap> 标签后检查
    clean_input = df['input_text'].str.replace(r'<gap>|<big_gap>', '', regex=True)
    clean_input = clean_input.str.replace('translate Akkadian to English:', '')
    
    combined_forbidden_in = "|".join(FORBIDDEN_INPUT_CHARS)
    mask_garbage_in = clean_input.str.contains(combined_forbidden_in, regex=True)
    fail_garbage_in = mask_garbage_in.sum()
    
    print(f"2. Input 端符号清理 (含点号 .):")
    if fail_garbage_in == 0:
        print(f"   ✅ PASS。Input 端纯净。")
    else:
        print(f"   ❌ FAIL! Input 端发现 {fail_garbage_in} 行残留符号 (如 . ! ? [ ] ( ))。")
        # 抽样展示
        if fail_garbage_in > 0:
            sample = clean_input[mask_garbage_in].iloc[0]
            print(f"      样例: ...{sample[-20:]}...")

    # [Check 3] Target 端符号清理 (English Mode)
    clean_target = df['target_text'].astype(str)
    combined_forbidden_out = "|".join(FORBIDDEN_TARGET_CHARS)
    mask_garbage_out = clean_target.str.contains(combined_forbidden_out, regex=True)
    fail_garbage_out = mask_garbage_out.sum()
    
    print(f"3. Target 端符号清理 (保留句号):")
    if fail_garbage_out == 0:
        print(f"   ✅ PASS。Target 端纯净。")
    else:
        print(f"   ❌ FAIL! Target 端发现 {fail_garbage_out} 行残留符号 (如 [ ] ( ) 等)。")

    # [Check 4] 下标归一化
    mask_sub = df['input_text'].str.contains(r'[₀-₉]', regex=True)
    fail_sub = mask_sub.sum()
    print(f"4. 下标数字归一化:")
    if fail_sub == 0:
        print(f"   ✅ PASS。无 Unicode 下标。")
    else:
        print(f"   ❌ FAIL! 发现 {fail_sub} 行残留下标。")

def audit_integrity(df):
    print_header("维度 2: 数据结构完整性 (Integrity)")
    
    n_rows = len(df)
    print(f"1. 数据量统计:")
    print(f"   - 总行数: {n_rows}")
    
    # [Check 1] 空值
    null_rows = df.isnull().any(axis=1).sum()
    
    # [Check 2] 换行符检测 (CSV Killer)
    newline_in = df['input_text'].str.contains(r'[\n\r]', regex=True).sum()
    newline_out = df['target_text'].str.contains(r'[\n\r]', regex=True).sum()
    
    print(f"2. 物理格式检查:")
    print(f"   - 空值行数: {null_rows}")
    print(f"   - Input 换行符: {newline_in}")
    print(f"   - Target 换行符: {newline_out}")
    
    if null_rows == 0 and newline_in == 0 and newline_out == 0:
        print(f"   ✅ PASS。CSV 格式安全。")
    else:
        print(f"   ❌ FAIL! 存在空值或换行符，会导致训练脚本崩溃。")

def audit_quality(df):
    print_header("维度 3: 数据内容质量 (Quality)")
    
    # [Check 1] URL 泄露检测
    url_leak = df['target_text'].astype(str).str.contains(URL_PATTERNS, regex=True, case=False).sum()
    print(f"1. URL 污染检测:")
    if url_leak == 0:
        print(f"   ✅ PASS。无网址残留。")
    else:
        print(f"   ❌ FAIL! 发现 {url_leak} 行包含 URL (http/search?q)。")
        
    # [Check 2] 极短输入
    short_rows = df[df['input_text'].str.len() < 35].shape[0]
    print(f"2. 极短样本检测 (<5 chars content):")
    print(f"   - 数量: {short_rows}")

def audit_linguistics(df):
    print_header("维度 4: 语言学特征 (Linguistics)")
    
    # [Check 1] 确定值风格
    has_curly = df['input_text'].str.contains(r'\{.*?\}', regex=True).sum()
    clean_text = df['input_text'].str.replace('translate Akkadian to English:', '')
    has_round = clean_text.str.contains(r'\(.*?\)', regex=True).sum()
    
    print(f"1. 确定值标记 ({{ }} vs ( )):")
    print(f"   - {{ }} 行数: {has_curly} (占比 {has_curly/len(df)*100:.1f}%)")
    print(f"   - ( ) 行数: {has_round}")
    
    if has_round > 0:
        print(f"   ❌ FAIL! 仍存在圆括号 ( )，未完成 V11 标准化。")
    else:
        print(f"   ✅ PASS。无圆括号残留。")

    # [Check 2] 阿卡德语字符
    has_akkadian = df['input_text'].str.contains(AKKADIAN_CHARS, regex=True).sum()
    print(f"2. 阿卡德语字符存活:")
    print(f"   - 包含变音符号行数: {has_akkadian}")

def audit_trainability(df):
    print_header("维度 5: 可训练性 (Trainability)")
    
    # [Check 1] Prefix
    prefix = "translate Akkadian to English: "
    has_prefix = df['input_text'].str.startswith(prefix).all()
    print(f"1. Task Prefix 检查:")
    if has_prefix:
        print(f"   ✅ PASS。")
    else:
        print(f"   ❌ FAIL! 前缀缺失。")
        
    # [Check 2] Length
    src_lens = df['input_text'].str.len()
    tgt_lens = df['target_text'].str.len()
    
    print(f"2. 序列长度统计:")
    print(f"   - Input  Max: {src_lens.max()}, Mean: {src_lens.mean():.1f}")
    print(f"   - Target Max: {tgt_lens.max()}, Mean: {tgt_lens.mean():.1f}")

    # [Check 3] Visual Inspection
    print(f"3. 最终人工抽样 (Random 3):")
    if len(df) > 0:
        sample = df.sample(min(3, len(df)))
        for i, (idx, row) in enumerate(sample.iterrows()):
            print(f"   --- Sample {i+1} ---")
            print(f"   IN : {str(row['input_text'])[:100]}...")
            print(f"   OUT: {str(row['target_text'])[:100]}...")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 致命错误: 找不到 {INPUT_FILE}")
        return
        
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    audit_compliance(df)
    audit_integrity(df)
    audit_quality(df)
    audit_linguistics(df)
    audit_trainability(df)
    
    print("\n" + "="*60)
    print(" 审计总结")
    print("="*60)
    print(f"目标文件: {INPUT_FILE}")
    print("验收标准: 所有关键项必须为 ✅ PASS，且 Target 端不得包含 URL。")

if __name__ == "__main__":
    main()