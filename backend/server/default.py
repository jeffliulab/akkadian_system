import hashlib
import random

# 字符池 (仅在该模块内部使用)
CUNEIFORM_POOL = [
    "𒀀", "𒀁", "𒀂", "𒀃", "𒀄", "𒀅", "𒀆", "𒀇", "𒀈", "𒀉", "𒀊", "𒀋", "𒀌", "𒀍", "𒀎",
    "ुलेंस", " ওষুধ", " পরিকল্পনা", "শিল্প", "垶", "জেল", "饪", " தயாரிக்க", "ਨੂੰ", "况", "ਕੰਮ"
]

def predict(text: str) -> str:
    """
    【Default Engine 具体实现】
    路径: server/model/model_default.py
    功能: 接收英文 -> MD5 Hashing -> 确定性楔形文字流
    """
    if not text:
        return ""
    
    # 1. 计算 Hashing
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # 2. 设定随机种子 (确保输入相同，输出永远相同)
    random.seed(text_hash)
    
    # 3. 生成逻辑
    length = min(len(text) * 2, 200) 
    result = []
    
    for i in range(length):
        char = random.choice(CUNEIFORM_POOL)
        result.append(char)
        # 随机插入空格
        if (i + 1) % random.randint(3, 8) == 0:
            result.append(" ")
            
    return "".join(result)