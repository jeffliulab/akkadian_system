"""
Transformer推理和翻译工具

整合自transformer/Translator.py
包含Beam Search解码实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamSearchTranslator(nn.Module):
    """
    使用Beam Search的翻译器
    
    Beam Search是一种启发式搜索算法，在每步保留k个最优候选序列，
    相比贪心解码能得到更好的翻译质量。
    
    使用示例:
        from new_transformer_model import Transformer
        
        # 加载训练好的模型
        model = Transformer(...)
        model.load_state_dict(torch.load('model.pth'))
        
        # 创建翻译器
        translator = BeamSearchTranslator(
            model=model,
            beam_size=5,
            max_seq_len=100,
            src_pad_idx=0,
            trg_pad_idx=0,
            trg_bos_idx=2,
            trg_eos_idx=3
        )
        
        # 翻译单个句子
        src_seq = torch.LongTensor([[1, 45, 23, 67, 3]]).to(device)
        translation = translator.translate_sentence(src_seq)
        print(translation)  # [2, 34, 56, 78, 3]
    """

    def __init__(self, model, beam_size, max_seq_len,
                 src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        """
        Args:
            model: 训练好的Transformer模型
            beam_size: Beam宽度 (推荐3-10，越大质量越好但速度越慢)
            max_seq_len: 最大生成长度
            src_pad_idx: 源语言padding索引
            trg_pad_idx: 目标语言padding索引
            trg_bos_idx: 目标语言开始标记索引
            trg_eos_idx: 目标语言结束标记索引
        """
        super(BeamSearchTranslator, self).__init__()

        self.alpha = 0.7  # 长度惩罚因子 (0.6-0.7较常用)
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        # 预分配buffer加速推理
        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        """
        解码器前向传播
        
        Args:
            trg_seq: 目标序列 (batch, seq_len)
            enc_output: 编码器输出
            src_mask: 源序列mask
            
        Returns:
            probs: 词汇表概率分布 (batch, seq_len, vocab_size)
        """
        from new_transformer_model import get_subsequent_mask
        
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_mask):
        """
        初始化Beam Search状态
        
        Args:
            src_seq: 源序列
            src_mask: 源序列mask
            
        Returns:
            enc_output: 编码器输出 (扩展到beam_size)
            gen_seq: 初始生成序列
            scores: 初始分数
        """
        beam_size = self.beam_size

        # 编码源序列
        enc_output, *_ = self.model.encoder(src_seq, src_mask)
        
        # 第一步解码，选择top-k个词作为beam起点
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        
        # 扩展编码器输出以匹配beam_size
        enc_output = enc_output.repeat(beam_size, 1, 1)
        
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        """
        在当前步选择最优的k个候选序列
        
        Args:
            gen_seq: 当前生成序列 (beam_size, max_seq_len)
            dec_output: 解码器输出概率
            scores: 当前累积分数 (beam_size,)
            step: 当前步数
            
        Returns:
            gen_seq: 更新后的生成序列
            scores: 更新后的分数
        """
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # 每个beam扩展k个候选，共k^2个候选
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # 加上之前的累积分数
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # 从k^2个候选中选出最优的k个
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # 计算这k个候选来自哪个beam和哪个词
        best_k_r_idxs = best_k_idx_in_k2 // beam_size  # beam索引
        best_k_c_idxs = best_k_idx_in_k2 % beam_size   # 词索引
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # 复制对应的历史序列
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # 设置当前步的词
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        """
        翻译单个句子
        
        Args:
            src_seq: 源序列 (1, src_len) - 注意batch_size必须为1
            
        Returns:
            list: 翻译结果的token id列表
        """
        # 当前实现只支持batch_size=1
        assert src_seq.size(0) == 1, "Beam Search当前只支持batch_size=1"

        src_pad_idx = self.src_pad_idx
        trg_eos_idx = self.trg_eos_idx
        max_seq_len = self.max_seq_len
        beam_size = self.beam_size
        alpha = self.alpha

        with torch.no_grad():
            from new_transformer_model import get_pad_mask
            
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # 默认选择第一个beam
            
            # 逐步生成，直到达到最大长度或所有beam都生成了EOS
            for step in range(2, max_seq_len):
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # 检查是否所有beam都已完成 (生成了EOS)
                eos_locs = gen_seq == trg_eos_idx
                
                # 用序列长度mask替换EOS位置
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                
                # 如果所有beam都包含EOS，则选择长度惩罚后分数最高的
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
            
            # 返回最优序列（截取到EOS）
            return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()


class GreedyTranslator:
    """
    贪心解码翻译器（速度快但质量略低于Beam Search）
    
    使用示例:
        translator = GreedyTranslator(model, max_len=100, bos_idx=2, eos_idx=3)
        translation = translator.translate(src_seq)
    """
    
    def __init__(self, model, max_len, bos_idx, eos_idx):
        """
        Args:
            model: Transformer模型
            max_len: 最大生成长度
            bos_idx: BOS token索引
            eos_idx: EOS token索引
        """
        self.model = model
        self.max_len = max_len
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.model.eval()
    
    @torch.no_grad()
    def translate(self, src_seq):
        """
        贪心解码翻译
        
        Args:
            src_seq: 源序列 (batch_size, src_len)
            
        Returns:
            生成的序列 (batch_size, trg_len)
        """
        device = src_seq.device
        batch_size = src_seq.size(0)
        
        # 初始化：所有句子都从BOS开始
        ys = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)
        
        for _ in range(self.max_len - 1):
            # 前向传播
            logits = self.model(src_seq, ys)
            logits = logits.view(batch_size, ys.size(1), -1)  # (B, L, vocab)
            
            # 贪心选择：取概率最大的词
            next_tok = logits[:, -1, :].argmax(-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            
            # 如果所有句子都生成了EOS，提前停止
            if (next_tok == self.eos_idx).all():
                break
        
        return ys


# ============================================================================
# 工具函数
# ============================================================================

def ids_to_sentence(ids, itos, bos_idx, eos_idx, pad_idx):
    """
    将token id序列转换为可读文本
    
    Args:
        ids: token id列表
        itos: id到字符串的映射字典 (list或dict)
        bos_idx: BOS索引
        eos_idx: EOS索引
        pad_idx: PAD索引
        
    Returns:
        str: 拼接后的句子
    """
    tokens = []
    for token_id in ids:
        if token_id == bos_idx:
            continue
        if token_id == eos_idx:
            break
        if token_id == pad_idx:
            continue
        tokens.append(itos[token_id])
    return " ".join(tokens)


def batch_translate(model, src_seqs, method='greedy', **kwargs):
    """
    批量翻译（支持greedy和beam search）
    
    Args:
        model: Transformer模型
        src_seqs: 源序列列表 (batch_size, src_len)
        method: 'greedy' 或 'beam'
        **kwargs: 翻译器参数
        
    Returns:
        list: 翻译结果列表
    """
    if method == 'greedy':
        translator = GreedyTranslator(model, **kwargs)
        return translator.translate(src_seqs).tolist()
    elif method == 'beam':
        translator = BeamSearchTranslator(model, **kwargs)
        # Beam search需要逐句处理
        results = []
        for src_seq in src_seqs:
            result = translator.translate_sentence(src_seq.unsqueeze(0))
            results.append(result)
        return results
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("="*60)
    print("Transformer推理模块")
    print("="*60)
    print("\n包含以下类:")
    print("  - BeamSearchTranslator: Beam Search解码器")
    print("  - GreedyTranslator: 贪心解码器")
    print("\n工具函数:")
    print("  - ids_to_sentence: ID序列转文本")
    print("  - batch_translate: 批量翻译")
    print("\n使用示例请参考文件顶部的docstring")
