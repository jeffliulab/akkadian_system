import os
import dill as pickle

def verify_dataset(data_dir="dataset"):
    print("="*50)
    print(f"Checking dataset in: {data_dir}")
    print("="*50)

    # 1. 检查文件是否存在
    files = ["train.src", "train.trg", "test.src", "akkadian.pkl"]
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"[ERROR] Missing file: {f}")
            return
        else:
            print(f"[OK] Found: {f} ({os.path.getsize(path)/1024:.1f} KB)")

    # 2. 验证行数对齐 (机器翻译的命脉)
    with open(os.path.join(data_dir, "train.src"), 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(os.path.join(data_dir, "train.trg"), 'r', encoding='utf-8') as f:
        trg_lines = f.readlines()

    if len(src_lines) != len(trg_lines):
        print(f"[CRITICAL ERROR] Line mismatch! src: {len(src_lines)}, trg: {len(trg_lines)}")
    else:
        print(f"[OK] Line count match: {len(src_lines)} pairs.")

    # 3. 抽样检查对齐质量 (检测是否还有长文档重复)
    print("\n" + "-"*30)
    print("Sampling first 3 pairs for manual check:")
    for i in range(min(3, len(src_lines))):
        print(f"Pair {i+1}:")
        print(f"  SRC (Akkadian): {src_lines[i][:100]}...")
        print(f"  TRG (English) : {trg_lines[i].strip()}")
    print("-" * 30)

    # 4. 验证词典结构 (底层 Transformer 必须)
    with open(os.path.join(data_dir, "akkadian.pkl"), "rb") as f:
        data = pickle.load(f)
        vocab = data.get("vocab", {})
        src_vocab = vocab.get("src", {}).get("itos", [])
        trg_vocab = vocab.get("trg", {}).get("itos", [])
        
        # 检查特殊 token 是否存在
        specials = ['<blank>', '<unk>', '<s>', '</s>']
        for s in specials:
            if s not in src_vocab:
                print(f"[ERROR] Special token {s} missing in SRC vocab!")
        
        print(f"[OK] Vocab validated. SRC: {len(src_vocab)}, TRG: {len(trg_vocab)}")

    # 5. 检查重复率 (如果重复率过高，说明切割逻辑失败)
    unique_src = len(set(src_lines))
    if unique_src < len(src_lines) * 0.5:
        print(f"[WARNING] High duplication! Only {unique_src} unique SRC lines out of {len(src_lines)}.")
    else:
        print(f"[OK] Source uniqueness check passed.")

if __name__ == "__main__":
    verify_dataset()