import torch
import torch.nn as nn
import pandas as pd
import dill as pickle
import os
import re

# 使用训练时相同的模型定义
from new_transformer_model import Transformer, PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD

# --- 配置参数 ---
class Config:
    data_pkl = 'dataset/akkadian.pkl'
    best_model = 'checkpoints_Transformer/best_model.pth'
    # 注意：模型超参数将从checkpoint中自动读取，无需手动设置！

def akkadian_clean(text):
    """
    与训练时完全一致的清洗逻辑
    参考: 2_preprocess_akkadian.py
    """
    if not isinstance(text, str):
        return ""
    # 统一 H
    text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    # 移除标注符号
    text = re.sub(r'[!?/:;˹˺]', '', text)
    # 处理gap标记
    text = text.replace('{large break}', '<big_gap>')
    text = re.sub(r'\[x\]', '<gap>', text)
    text = re.sub(r'\[\.\.\.\]', '<big_gap>', text)
    # 移除括号
    text = text.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    return text.strip().lower()

@torch.no_grad()
def greedy_decode(model, src_seq, bos_idx, eos_idx, trg_vocab_size, max_len=100):
    """通用贪心解码"""
    device = src_seq.device
    B = src_seq.size(0)
    ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        # model返回的是 (batch*seq_len, vocab_size)，需要reshape
        logits = model(src_seq, ys)
        logits = logits.view(B, ys.size(1), trg_vocab_size)  # (B, L, vocab)
        next_tok = logits[:, -1, :].argmax(-1)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        if (next_tok == eos_idx).all(): break
    return ys

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载checkpoint（从中读取所有配置）
    if not os.path.exists(Config.best_model):
        print(f"[Error] {Config.best_model} not found!")
        print("[Error] Please ensure the model file exists in the submit/ folder.")
        return
        
    print(f"[Info] Loading checkpoint: {Config.best_model}")
    checkpoint = torch.load(Config.best_model, map_location=device, weights_only=False)
    
    # 2. 从checkpoint读取配置
    if 'hp_cont' not in checkpoint:
        print("[Error] Checkpoint does not contain hp_cont! Cannot reconstruct model.")
        print("[Error] Please retrain the model with the updated training script.")
        return
    
    hp = checkpoint['hp_cont']
    print("[Info] ✓ Configuration loaded from checkpoint:")
    print(f"  - d_model={hp['d_model']}, n_layers={hp['n_layers']}, n_head={hp['n_head']}")
    print(f"  - dropout={hp.get('dropout', 0.0)}, norm_inside_residual={hp.get('norm_inside_residual', False)}")
    
    # 3. 加载词典
    with open(Config.data_pkl, 'rb') as f:
        data = pickle.load(f)
    vocab = data['vocab']
    src_stoi = vocab['src']['stoi']
    trg_stoi = vocab['trg']['stoi']
    trg_itos = vocab['trg']['itos']
    
    # 获取特殊词 ID
    src_pad_idx = src_stoi[PAD_WORD]
    src_bos_idx = src_stoi[BOS_WORD]
    src_eos_idx = src_stoi[EOS_WORD]
    src_unk_idx = src_stoi[UNK_WORD]
    
    trg_pad_idx = trg_stoi[PAD_WORD]
    trg_bos_idx = trg_stoi[BOS_WORD]
    trg_eos_idx = trg_stoi[EOS_WORD]
    trg_unk_idx = trg_stoi[UNK_WORD]

    # 4. 使用checkpoint中的配置初始化模型
    n_trg_vocab = len(trg_itos)
    
    model = Transformer(
        n_src_vocab=len(vocab['src']['itos']),
        n_trg_vocab=n_trg_vocab,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        d_k=hp['d_k'],
        d_v=hp['d_v'],
        d_model=hp['d_model'],
        d_word_vec=hp['d_word_vec'],
        d_inner=hp['d_inner_hid'],
        n_layers=hp['n_layers'],
        n_head=hp['n_head'],
        dropout=0.0,  # 推理时关闭dropout
        n_position=hp.get('n_position', 200),
        trg_emb_prj_weight_sharing=hp.get('trg_emb_prj_weight_sharing', True),
        emb_src_trg_weight_sharing=hp.get('emb_src_trg_weight_sharing', True),
        scale_emb_or_prj=hp.get('scale_emb_or_prj', 'emb'),
        norm_inside_residual=hp.get('norm_inside_residual', False),
        norm_class=nn.LayerNorm
    ).to(device)

    # 5. 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[Info] Loaded from 'model_state_dict' key")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("[Info] Loaded from 'model' key")
    else:
        # 如果checkpoint就是state_dict本身
        model.load_state_dict(checkpoint)
        print("[Info] Loaded checkpoint directly as state_dict")
    
    model.eval()
    print("[Info] Model loaded successfully!")

    # 4. 读取测试集转写
    test_src_path = 'dataset/test.src'
    with open(test_src_path, 'r', encoding='utf-8') as f:
        test_lines = [ln.strip() for ln in f]

    print(f"[Info] Ready to translate {len(test_lines)} sentences on {device}...")
    results = []

    for idx, raw_text in enumerate(test_lines):
        # 与训练时一致的预处理！
        cleaned_text = akkadian_clean(raw_text)
        tokens = cleaned_text.split()
        
        # 转换并添加边界符（使用源语言索引）
        ids = [src_bos_idx] + [src_stoi.get(w, src_unk_idx) for w in tokens] + [src_eos_idx]
        src_seq = torch.tensor([ids], dtype=torch.long).to(device)
        
        # 解码（使用目标语言索引）
        gen_ids = greedy_decode(model, src_seq, trg_bos_idx, trg_eos_idx, n_trg_vocab)
        
        # 转换回文本（使用目标语言索引）
        out_sent = []
        for tid in gen_ids[0].tolist():
            if tid in [trg_bos_idx, trg_pad_idx]:
                continue
            if tid == trg_eos_idx:
                break
            out_sent.append(trg_itos[tid])
        
        translation = " ".join(out_sent) if out_sent else "empty translation"
        results.append(translation)
        
        # 进度显示
        if (idx + 1) % 100 == 0:
            print(f"  Translated {idx + 1}/{len(test_lines)} sentences...")

    # 5. 生成最终提交文件
    test_csv_path = 'deep-past-initiative-machine-translation/test.csv'
    if not os.path.exists(test_csv_path):
        print(f"[Error] {test_csv_path} not found!")
        print("[Info] Creating dummy submission with sequential IDs...")
        submission = pd.DataFrame({
            'id': range(len(results)),
            'translation': results
        })
    else:
        raw_test_df = pd.read_csv(test_csv_path)
        if len(results) != len(raw_test_df):
            print(f"[Warning] Result count ({len(results)}) != test.csv rows ({len(raw_test_df)})")
            print("[Info] Adjusting...")
            # 补齐或截断
            while len(results) < len(raw_test_df):
                results.append("empty translation")
            results = results[:len(raw_test_df)]
        
        submission = pd.DataFrame({'id': raw_test_df['id'], 'translation': results})
    
    submission.to_csv('submission.csv', index=False)
    print(f"[Success] submission.csv created with {len(submission)} translations!")
    print(f"[Info] Sample translations:")
    for i in range(min(3, len(submission))):
        print(f"  [{i}] {submission.iloc[i]['translation'][:80]}...")

if __name__ == "__main__":
    main()