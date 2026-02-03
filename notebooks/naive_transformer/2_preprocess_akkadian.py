import os
import pandas as pd
import re
import dill as pickle
from collections import Counter

# --- 常量定义 ---
class _C:
    PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD = '<blank>', '<unk>', '<s>', '</s>'
Constants = _C()

def akkadian_clean(text):
    """针对转写的核心清洗逻辑"""
    if not isinstance(text, str): return ""
    text = text.replace('ḫ', 'h').replace('Ḫ', 'H') # 统一 H
    text = re.sub(r'[!?/:;˹˺]', '', text) # 移除标注
    text = text.replace('{large break}', '<big_gap>')
    text = re.sub(r'\[x\]', '<gap>', text)
    text = re.sub(r'\[\.\.\.\]', '<big_gap>', text)
    text = text.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    return text.strip().lower()

def split_src_by_anchors(full_text, sentence_group):
    """
    利用句首词锚点将长文档切分为短句
    """
    # 按照文档中的顺序排序
    group = sentence_group.sort_values('sentence_obj_in_text')
    anchors = group['first_word_spelling'].fillna('').tolist()
    translations = group['translation'].fillna('').tolist()
    
    sentence_srcs = []
    text_ptr = 0
    
    for i in range(len(anchors)):
        current_anchor = str(anchors[i]).strip().lower()
        next_anchor = str(anchors[i+1]).strip().lower() if i+1 < len(anchors) else None
        
        # 寻找当前锚点在全文中的位置
        start_pos = full_text.lower().find(current_anchor, text_ptr)
        if start_pos == -1: start_pos = text_ptr
        
        # 寻找下一个锚点的位置作为结束点
        if next_anchor:
            end_pos = full_text.lower().find(next_anchor, start_pos + len(current_anchor))
            if end_pos == -1: end_pos = len(full_text)
        else:
            end_pos = len(full_text)
            
        # 切出这一句对应的阿卡德语
        segment = full_text[start_pos:end_pos].strip()
        sentence_srcs.append(akkadian_clean(segment))
        text_ptr = end_pos
        
    return sentence_srcs, [str(t).strip().lower() for t in translations]

def main():
    data_dir = "../../data/deep-past-initiative-machine-translation"
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    align_df = pd.read_csv(os.path.join(data_dir, 'Sentences_Oare_FirstWord_LinNum.csv'))

    all_src, all_trg = [], []

    # 2. 句子级精准对齐
    print("[Info] Cutting documents into sentences using anchors...")
    for text_uuid, group in align_df.groupby('text_uuid'):
        doc_match = train_df[train_df['oare_id'] == text_uuid]
        if doc_match.empty: continue
        
        full_transliteration = doc_match.iloc[0]['transliteration']
        
        # 执行物理切分
        src_list, trg_list = split_src_by_anchors(full_transliteration, group)
        
        for s, t in zip(src_list, trg_list):
            if len(t.split()) > 1 and len(s) > 0: # 过滤无效行
                all_src.append(s)
                all_trg.append(t)

    # 3. 处理测试集（测试集已经是句子级，直接清洗）
    test_src = test_df['transliteration'].apply(akkadian_clean).tolist()

    # 4. 导出结果
    print(f"[Success] Final aligned pairs: {len(all_src)}")
    with open(os.path.join(output_dir, "train.src"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_src))
    with open(os.path.join(output_dir, "train.trg"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_trg))
    with open(os.path.join(output_dir, "test.src"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_src))

    # 5. 生成词典 (Shared Vocab 模式)
    def build_v(data_list):
        counter = Counter()
        for ln in data_list: counter.update(ln.split())
        itos = [Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD] + \
               [w for w, c in counter.items() if c >= 2]
        return {"itos": itos, "stoi": {w: i for i, w in enumerate(itos)}}

    vocab = {"src": build_v(all_src), "trg": build_v(all_trg)}
    with open(os.path.join(output_dir, "akkadian.pkl"), "wb") as f:
        pickle.dump({"vocab": vocab}, f)
    
    print(f"[Info] Preprocessing complete. Vocab size: {len(vocab['src']['itos'])}")

if __name__ == "__main__":
    main()