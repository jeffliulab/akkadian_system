import pandas as pd
import os
import re
from pathlib import Path

# ================= 配置区 =================
CONFIG = {
    "data_dir": "../../data/deep-past-initiative-machine-translation",
    "files": {
        "train": "train.csv",
        "sentences": "Sentences_Oare_FirstWord_LinNum.csv",
        "published": "published_texts.csv"
    },
    "output_file": "clean.csv"  # <--- 已修改
}

class SourceNormalizer:
    """阿卡德语输入端清洗器"""
    def __init__(self):
        self.sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    def normalize(self, text: str, convert_parens: bool = False) -> str:
        if not isinstance(text, str) or len(text) < 2: return ""
        
        # 1. 字符统一
        text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
        
        # 2. 括号转换 (针对 train.csv 的 (d) -> {d})
        if convert_parens:
            text = text.replace('(', '{').replace(')', '}')
        
        # 3. 保护逻辑
        text = text.replace('[... ...]', '@BIGGAP@').replace('...', '@BIGGAP@').replace('[x]', '@GAP@')
        text = re.sub(r'\{(.*?)\}', r'@DET_\1@', text)
        
        # 4. 下标转换
        text = text.translate(self.sub_map)
        
        # 5. 暴力除杂 (阿卡德语端移除点号)
        # 移除 ! ? : ; [ ] ( ) / \ < > .
        forbidden_pattern = r'[!?:;\[\]\(\)˹˺/\\<>\.]' 
        text = re.sub(forbidden_pattern, '', text)
        
        # 6. 还原保护
        text = text.replace('@BIGGAP@', '<big_gap>').replace('@GAP@', '<gap>')
        text = re.sub(r'@DET_(.*?)@', r'{\1}', text)
        
        # 7. 压缩空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class TargetNormalizer:
    """英语输出端清洗器"""
    def normalize(self, text: str) -> str:
        if not isinstance(text, str): return ""
        
        # 1. 移除考古标注 (保留句号 . 和逗号 ,)
        # 移除 ! ? / < > [ ] ( )
        text = re.sub(r'[!?:;\[\]\(\)˹˺/\\<>]', '', text)
        
        # 2. 移除换行符 (关键：防止 CSV 错行)
        text = text.replace('\n', ' ').replace('\r', '')
        
        # 3. 压缩空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def is_valid_translation(text: str) -> bool:
    """脏数据过滤器"""
    if not isinstance(text, str): return False
    text = text.strip()
    if len(text) < 2: return False
    if text.lower() == 'nan': return False
    
    # 拦截 URL
    if "http" in text or "search?q=" in text or "www." in text:
        return False
        
    # 拦截无字母的垃圾
    if not re.search(r'[a-zA-Z]', text):
        return False
    return True

def process_track_a_sentences(train_df, sent_df, src_norm, tgt_norm):
    print(f"  [Track A] 启动锚点对齐...")
    aligned = []
    sent_grouped = sent_df.groupby('text_uuid')
    
    for doc_id, group in sent_grouped:
        doc_rows = train_df[train_df['oare_id'] == doc_id]
        if doc_rows.empty: continue
        full_text = str(doc_rows.iloc[0]['transliteration'])
        
        cursor = 0
        sorted_sents = group.sort_values('sentence_obj_in_text')
        
        anchors = sorted_sents['first_word_spelling'].tolist()
        trans = sorted_sents['translation'].tolist()
        
        for i, anchor in enumerate(anchors):
            anchor_s = str(anchor)
            target_s = str(trans[i])
            
            if pd.isna(anchor) or not is_valid_translation(target_s): continue
            
            start = full_text.find(anchor_s, cursor)
            if start == -1: continue
            
            end = len(full_text)
            if i + 1 < len(anchors):
                next_anchor = str(anchors[i+1])
                next_pos = full_text.find(next_anchor, start + len(anchor_s))
                if next_pos != -1: end = next_pos
            
            # === 双端清洗 ===
            raw_seg = full_text[start:end]
            clean_in = src_norm.normalize(raw_seg, convert_parens=True)
            clean_out = tgt_norm.normalize(target_s)
            
            if clean_in and clean_out:
                aligned.append({
                    "input_text": f"translate Akkadian to English: {clean_in}", 
                    "target_text": clean_out, 
                    "source": "train_anchor"
                })
            cursor = end
            
    print(f"  [Track A] 有效产出: {len(aligned)}")
    return aligned

def process_track_b_published(pub_df, src_norm, tgt_norm):
    print(f"  [Track B] 启动 URL 过滤...")
    aligned = []
    
    valid_pub = pub_df.dropna(subset=['transliteration', 'AICC_translation'])
    
    for _, row in valid_pub.iterrows():
        raw_in = str(row['transliteration'])
        raw_out = str(row['AICC_translation'])
        
        if not is_valid_translation(raw_out): continue
        if len(raw_in) < 5: continue
        
        # === 双端清洗 ===
        clean_in = src_norm.normalize(raw_in, convert_parens=False)
        clean_out = tgt_norm.normalize(raw_out)
        
        if clean_in and clean_out:
            aligned.append({
                "input_text": f"translate Akkadian to English: {clean_in}", 
                "target_text": clean_out, 
                "source": "published_text"
            })
        
    print(f"  [Track B] 有效产出: {len(aligned)}")
    return aligned

def main():
    print(f"{'='*30} Data Factory (Output: clean.csv) {'='*30}")
    base = Path(CONFIG["data_dir"])
    src_norm = SourceNormalizer()
    tgt_norm = TargetNormalizer()
    all_data = []
    
    try:
        df_train = pd.read_csv(base / CONFIG["files"]["train"], encoding='utf-8')
        df_sent = pd.read_csv(base / CONFIG["files"]["sentences"], encoding='utf-8')
        df_pub = pd.read_csv(base / CONFIG["files"]["published"], encoding='utf-8')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 执行
    all_data.extend(process_track_a_sentences(df_train, df_sent, src_norm, tgt_norm))
    all_data.extend(process_track_b_published(df_pub, src_norm, tgt_norm))
    
    final_df = pd.DataFrame(all_data)
    
    if final_df.empty:
        print("❌ 错误: 0 样本")
        return

    # 去重
    final_df = final_df.drop_duplicates(subset=['input_text'])
    
    # 导出
    final_df.to_csv(CONFIG["output_file"], index=False, encoding='utf-8')
    
    print(f"\n{'='*30} 完成 {'='*30}")
    print(f"  - 总样本数: {len(final_df)}")
    print(f"  - 输出文件: {CONFIG['output_file']}")
    print(f"  - 状态: 已准备好进行审计")

if __name__ == "__main__":
    main()