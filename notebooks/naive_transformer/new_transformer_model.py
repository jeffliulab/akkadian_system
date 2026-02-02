"""
完整的Transformer模型实现
整合自transformer文件夹的所有核心组件

包含:
- Constants: 特殊token定义
- Modules: ScaledDotProductAttention
- SubLayers: MultiHeadAttention, PositionwiseFeedForward
- Layers: EncoderLayer, DecoderLayer
- Models: Transformer主模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# CONSTANTS - 特殊Token定义
# ============================================================================

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


# ============================================================================
# MODULES - 基础注意力机制
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch, n_head, seq_len, d_k/d_v)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


# ============================================================================
# SUBLAYERS - 子层组件
# ============================================================================

class MultiHeadAttention(nn.Module):
    """多头注意力模块"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, 
                 norm_inside_residual=False, norm_class=None):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.norm_inside_residual = norm_inside_residual
        self.norm = norm_class(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        
        # Pre-normalization (如果启用)
        if self.norm_inside_residual:
            q = self.norm(q)
            k = self.norm(k)
            v = self.norm(v)

        # 投影到多头: b x lq x (n*dv)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # 转置用于注意力计算: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # 广播到多头

        q, attn = self.attention(q, k, v, mask=mask)

        # 转回并合并多头: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        # 残差连接
        q += residual

        # Post-normalization (默认)
        if not self.norm_inside_residual:
            q = self.norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络 (FFN)"""

    def __init__(self, d_in, d_hid, dropout=0.1, 
                 norm_inside_residual=False, norm_class=None):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.norm = norm_class(d_in, eps=1e-6)
        self.norm_inside_residual = norm_inside_residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        
        # Pre-normalization (如果启用)
        if self.norm_inside_residual:
            x = self.norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        # 残差连接
        x += residual
        
        # Post-normalization (默认)
        if not self.norm_inside_residual:
            x = self.norm(x)

        return x


# ============================================================================
# LAYERS - 编码器/解码器层
# ============================================================================

class EncoderLayer(nn.Module):
    """Transformer编码器层 (Self-Attention + FFN)"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, 
                 norm_inside_residual=False, norm_class=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, 
            norm_inside_residual=norm_inside_residual, norm_class=norm_class)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, 
            norm_inside_residual=norm_inside_residual, norm_class=norm_class)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """Transformer解码器层 (Self-Attention + Cross-Attention + FFN)"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, 
                 norm_inside_residual=False, norm_class=None):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, 
            norm_inside_residual=norm_inside_residual, norm_class=norm_class)
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, 
            norm_inside_residual=norm_inside_residual, norm_class=norm_class)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, 
            norm_inside_residual=norm_inside_residual, norm_class=norm_class)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


# ============================================================================
# MODELS - 完整Transformer模型
# ============================================================================

def get_pad_mask(seq, pad_idx):
    """生成padding mask"""
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """生成因果mask (用于解码器自注意力)"""
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """生成正弦位置编码表"""
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=200, 
                 scale_emb=False, norm_inside_residual=False, norm_class=None):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, 
                        norm_inside_residual=norm_inside_residual, norm_class=norm_class)
            for _ in range(n_layers)])
        self.layer_norm = norm_class(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        # Embedding + Positional Encoding
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        # 堆叠编码器层
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, n_position=200, dropout=0.1, 
                 scale_emb=False, norm_inside_residual=False, norm_class=None):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, 
                        norm_inside_residual=norm_inside_residual, norm_class=norm_class)
            for _ in range(n_layers)])
        self.norm = norm_class(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # Embedding + Positional Encoding
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.norm(dec_output)

        # 堆叠解码器层
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    """
    完整的Transformer模型 (Encoder-Decoder架构)
    
    参数说明:
        n_src_vocab: 源语言词汇表大小
        n_trg_vocab: 目标语言词汇表大小
        src_pad_idx: 源语言padding索引
        trg_pad_idx: 目标语言padding索引
        d_word_vec: 词向量维度 (默认512)
        d_model: 模型维度 (默认512)
        d_inner: FFN内部维度 (默认2048)
        n_layers: 编码器/解码器层数 (默认6)
        n_head: 注意力头数 (默认8)
        d_k, d_v: 每个头的key/value维度 (默认64)
        dropout: Dropout比例 (默认0.1)
        n_position: 最大序列长度 (默认200)
        trg_emb_prj_weight_sharing: 目标embedding和输出层权重共享
        emb_src_trg_weight_sharing: 源和目标embedding权重共享
        scale_emb_or_prj: 缩放策略 ('emb'/'prj'/'none')
        norm_inside_residual: 使用Pre-LN还是Post-LN (True=Pre-LN)
        norm_class: 归一化类 (nn.LayerNorm或自定义)
    """

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj', norm_inside_residual=False, norm_class=None):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # 缩放策略 (参考"Attention Is All You Need"论文3.4节)
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        # 编码器
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb,
            norm_inside_residual=norm_inside_residual, norm_class=norm_class)

        # 解码器
        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb,
            norm_inside_residual=norm_inside_residual, norm_class=norm_class)

        # 输出投影层
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        # Xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
            '为了使用残差连接，所有模块输出维度必须相同'

        # 权重共享
        if trg_emb_prj_weight_sharing:
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        """
        前向传播
        
        Args:
            src_seq: 源序列 (batch_size, src_len)
            trg_seq: 目标序列 (batch_size, trg_len)
            
        Returns:
            logits: 词汇表上的logits (batch_size * trg_len, n_trg_vocab)
        """
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建一个小型Transformer用于测试
    model = Transformer(
        n_src_vocab=1000,
        n_trg_vocab=1000,
        src_pad_idx=0,
        trg_pad_idx=0,
        d_word_vec=256,
        d_model=256,
        d_inner=1024,
        n_layers=4,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        norm_class=nn.LayerNorm
    )
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    src = torch.randint(1, 100, (2, 10))
    trg = torch.randint(1, 100, (2, 8))
    
    logits = model(src, trg)
    print(f"输出形状: {logits.shape}  # 应该是 (batch*seq_len, vocab_size)")
