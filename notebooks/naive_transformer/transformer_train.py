import math
import time
import dill as pickle
import numpy as np
import random
import os
import sacrebleu 

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from new_transformer_model import Transformer, PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD
from new_transformer_optim import ScheduledOptim

from checkpoint import Checkpointer
from types import SimpleNamespace
from typing import List, Tuple
from torch.utils.data import DataLoader

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        return F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum', label_smoothing=eps)
    else:
        return F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')

def patch_src(src, pad_idx):
    # loader yields [S, B]; model expects [B, S]
    return src.transpose(0, 1)

def patch_trg(trg, pad_idx):
    # [T, B] -> [B, T]; split into teacher-forcing input/targets
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def ids_to_sentence(ids, itos, bos_idx, eos_idx, pad_idx):
    s = []
    for t in ids:
        if t == bos_idx: 
            continue
        if t == eos_idx:
            break
        if t == pad_idx:
            continue
        s.append(itos[t])
    return " ".join(s)

@torch.no_grad()
def greedy_decode(model, src_seq, bos_idx, eos_idx, max_len):
    device = src_seq.device
    B = src_seq.size(0)
    ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits = model(src_seq, ys)
        logits = logits.view(src_seq.size(0), ys.size(1), -1)  # (B, L, vocab)
        next_tok = logits[:, -1, :].argmax(-1)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        if (next_tok == eos_idx).all():
            break
    return ys

PAD, UNK, BOS, EOS = PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD

def _read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _tok_lower(s: str) -> List[str]:
    return s.strip().lower().split()

def _add_bos_eos(tokens: List[str]) -> List[str]:
    return [BOS] + tokens + [EOS]

def _to_ids(tokens: List[str], stoi: dict) -> List[int]:
    unk = stoi[UNK]
    return [stoi.get(w, unk) for w in tokens]

class _Batch:
    """Mimics torchtext Batch: .src and .trg are LongTensors shaped [seq_len, batch]."""
    __slots__ = ("src", "trg")
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

def _make_pairs(src_path: str, trg_path: str, max_len: int):
    src_lines = _read_lines(src_path)
    trg_lines = _read_lines(trg_path)
    pairs = []
    for s, t in zip(src_lines, trg_lines):
        s_tok, t_tok = _tok_lower(s), _tok_lower(t)
        if len(s_tok) <= max_len and len(t_tok) <= max_len:
            pairs.append((s_tok, t_tok))
    return pairs

def _collate_pairs(pairs, src_stoi, trg_stoi, src_pad_idx, trg_pad_idx):
    src_tok = [_add_bos_eos(s) for s, _ in pairs]
    trg_tok = [_add_bos_eos(t) for _, t in pairs]
    # map to ids - 使用各自的stoi！
    src_ids = [_to_ids(s, src_stoi) for s in src_tok]
    trg_ids = [_to_ids(t, trg_stoi) for t in trg_tok]

    # dynamic pad -> time-major [L, B] - 使用各自的pad_idx！
    Ls = max(len(s) for s in src_ids)
    Lt = max(len(t) for t in trg_ids)
    B = len(src_ids)
    src = torch.full((Ls, B), src_pad_idx, dtype=torch.long)
    trg = torch.full((Lt, B), trg_pad_idx, dtype=torch.long)
    for b, s in enumerate(src_ids):
        src[:len(s), b] = torch.tensor(s, dtype=torch.long)
    for b, t in enumerate(trg_ids):
        trg[:len(t), b] = torch.tensor(t, dtype=torch.long)
    return _Batch(src, trg)

def init_transformer_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)

def prepare_dataloaders(hp_cont, device):
    # 加载我们新生成的 akkadian.pkl
    data = pickle.load(open(hp_cont.data_pkl, 'rb'))
    vocab = data['vocab']
    
    # 适配新的词典结构
    hp_cont.stoi = vocab['src']['stoi']
    hp_cont.itos = vocab['src']['itos']
    hp_cont.trg_stoi = vocab['trg']['stoi']
    hp_cont.trg_itos = vocab['trg']['itos']

    hp_cont.src_pad_idx = hp_cont.stoi[PAD]
    hp_cont.trg_pad_idx = hp_cont.trg_stoi[PAD]
    hp_cont.src_vocab_size = len(hp_cont.itos)
    hp_cont.trg_vocab_size = len(hp_cont.trg_itos)
    
    hp_cont.bos_idx = hp_cont.stoi[BOS]
    hp_cont.eos_idx = hp_cont.stoi[EOS]

    # 直接读取我们生成的 .src 和 .trg 文件路径
    train_src, train_trg = 'dataset/train.src', 'dataset/train.trg'
    test_src = 'dataset/test.src' # 注意：测试集没给标签，这里可能需要逻辑适配

    train_pairs = _make_pairs(train_src, train_trg, hp_cont.max_token_seq_len)
    # 暂时用 train 的一部分做验证集，因为原始数据没给验证集
    val_pairs = train_pairs[-100:] 
    train_pairs = train_pairs[:-100]

    from functools import partial
    collate_fn = partial(_collate_pairs, 
                        src_stoi=hp_cont.stoi, 
                        trg_stoi=hp_cont.trg_stoi,
                        src_pad_idx=hp_cont.src_pad_idx,
                        trg_pad_idx=hp_cont.trg_pad_idx)
    
    train_loader = DataLoader(train_pairs, batch_size=hp_cont.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_pairs, batch_size=hp_cont.batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, None # 测试集逻辑建议单独写预测脚本


def train_epoch(model, training_data, optimizer, hp_cont, device, warm_up_bool):
    ''' Epoch operation in training phase'''
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    for batch in training_data:
        src_seq = patch_src(batch.src, hp_cont.src_pad_idx).to(device, non_blocking=True)
        trg_seq, gold = map(lambda x: x.to(device, non_blocking=True), patch_trg(batch.trg, hp_cont.trg_pad_idx))
        if warm_up_bool:
            optimizer.zero_grad()
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(pred, gold, hp_cont.trg_pad_idx, smoothing=hp_cont.label_smoothing)
            loss.backward()
            optimizer.step_and_update_lr()
        else:
            optimizer.zero_grad()
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(pred, gold, hp_cont.trg_pad_idx, smoothing=hp_cont.label_smoothing)
            loss.backward()
            optimizer.step()
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, hp_cont):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    total_loss, n_word_total = 0, 0
    sys_out, ref_out = [], []

    with torch.no_grad():
        for batch in validation_data:
            src_seq = patch_src(batch.src, hp_cont.src_pad_idx).to(device, non_blocking=True)
            trg_seq, gold = map(lambda x: x.to(device, non_blocking=True), patch_trg(batch.trg, hp_cont.trg_pad_idx))
            pred = model(src_seq, trg_seq)
            loss, _, n_word = cal_performance(pred, gold, hp_cont.trg_pad_idx, smoothing=False)
            n_word_total += n_word
            total_loss += loss.item()
            
            # greedy decode
            gen = greedy_decode(model, src_seq, hp_cont.bos_idx, hp_cont.eos_idx, hp_cont.max_token_seq_len)
            ref = batch.trg.transpose(0, 1)
            for h_ids, r_ids in zip(gen.tolist(), ref.tolist()):
                sys_out.append(ids_to_sentence(h_ids, hp_cont.trg_itos, hp_cont.bos_idx, hp_cont.eos_idx, hp_cont.trg_pad_idx))
                ref_out.append(ids_to_sentence(r_ids, hp_cont.trg_itos, hp_cont.bos_idx, hp_cont.eos_idx, hp_cont.trg_pad_idx))
                
    loss_per_word = total_loss / n_word_total
    bleu = sacrebleu.corpus_bleu(sys_out, [ref_out]).score  # 0-100
    return loss_per_word, bleu

#===================================================================================================================================
# MODIFY YOUR CODES HERE: REGULARIZATION
# For RAdam, please use the RAdam package we imported (the external implementation), not the one from torch.optim.
def make_regularization(reg_name: str = "no_reg", wd: float = 1e-2, lr: float = 1e-4, betas=(0.9, 0.98), eps: float = 1e-9, params=None):
    """
    Returns:
        optimizer (torch.optim.Optimizer or None): if params is None -> None
    """
    name = reg_name.lower()
    opt_inst = None
    if params is None:
        return None
    if name == "no_reg":
        opt_inst = optim.Adam(params, lr=lr, betas=betas, eps=eps)
    elif name == "l2":
        opt_inst = optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    elif name == "weight_decay":
        opt_inst = optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    elif name =="rectified":
        # Dynamically import RAdam when needed
        from radam import RAdam
        opt_inst = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    else:
        raise ValueError(f"Unknown reg_name: {reg_name!r}. Use 'no_reg', 'l2', 'weight_decay', or 'rectified'.")
    return opt_inst
#===================================================================================================================================


#===================================================================================================================================
# MODIFY YOUR CODES HERE: NORMALIZATION
# The transformer input shape is (B, L, d_model); transpose the input to match nn.BatchNorm1d's expected input shape (B, d_model, L). Don't forget to transpose it back to the input shape.
def make_normalization(norm_name: str = "layernorm"):
    """
    Returns:
        a normalization class to be passed into Transformer(norm_class=...)
        - 'layernorm'  -> nn.LayerNorm
        - 'batchnorm'  -> nn.BatchNorm1d
    """
    name = norm_name.lower()
    if name=="layernorm":
        return nn.LayerNorm
    elif name=="batchnorm":
        # Wrapper class for BatchNorm1d to handle (B, L, d_model) -> (B, d_model, L) transpose
        class BatchNorm1dWrapper(nn.Module):
            def __init__(self, normalized_shape, eps=1e-6):
                super().__init__()
                self.bn = nn.BatchNorm1d(normalized_shape, eps=eps)
            
            def forward(self, x):
                # x: (B, L, d_model)
                # BatchNorm1d expects (B, d_model, L)
                x = x.transpose(1, 2)  # (B, d_model, L)
                x = self.bn(x)
                x = x.transpose(1, 2)  # (B, L, d_model)
                return x
        return BatchNorm1dWrapper
    else:
        raise ValueError(f"Unknown norm_name: {norm_name!r}. Use 'layernorm' or 'batchnorm'.")
#===================================================================================================================================



def main(ckpt, device, hp_cont, dropout_bool=True, reg_name="no_reg", norm_name="layernorm", norm_inside_residual=False, warm_up_bool=True):
    
#===================================================================================================================================
# MODIFY YOUR CODES HERE: DROPOUT
    hp_cont.dropout = 0.1 if dropout_bool else 0.0
#===================================================================================================================================
    hp_cont.n_warmup_steps = 4000
    hp_cont.norm_inside_residual = norm_inside_residual  # 关键修复：保存到checkpoint
    norm_class = make_normalization(norm_name=norm_name)
    
    # --- seed: set random seeds for reproducibility ---
    torch.manual_seed(hp_cont.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hp_cont.seed)
        
    # --- data ---
    training_data, validation_data, test_data = prepare_dataloaders(hp_cont, device)

    # --- model ---
    model = Transformer(
        hp_cont.src_vocab_size,
        hp_cont.trg_vocab_size,
        src_pad_idx=hp_cont.src_pad_idx,
        trg_pad_idx=hp_cont.trg_pad_idx,
        trg_emb_prj_weight_sharing=hp_cont.proj_share_weight,
        emb_src_trg_weight_sharing=hp_cont.embs_share_weight,
        d_k=hp_cont.d_k,
        d_v=hp_cont.d_v,
        d_model=hp_cont.d_model,
        d_word_vec=hp_cont.d_word_vec,
        d_inner=hp_cont.d_inner_hid,
        n_layers=hp_cont.n_layers,
        n_head=hp_cont.n_head,
        scale_emb_or_prj=hp_cont.scale_emb_or_prj,
        dropout=hp_cont.dropout,
        norm_inside_residual=norm_inside_residual,
        norm_class=norm_class
    ).to(device)
    
    # --- initialization ---
    # 注释掉：new_transformer_model已经在__init__中初始化了
    # 重复初始化会破坏权重共享！
    # model.apply(init_transformer_weights)
    
    # --- regularization ---
    opt_inst = make_regularization(reg_name=reg_name, wd=hp_cont.wd, lr=hp_cont.lr, betas=hp_cont.betas, eps=1e-9, params=model.parameters())
    
    # --- optimizer / scheduler ---
   #===================================================================================================================================
# MODIFY YOUR CODES HERE: LEARNING-RATE WARM-UP
    if warm_up_bool:
        # 修复：对于小模型(d_model<512)，需要学习率缩放
        # d_model=256时，base_lr=1/sqrt(256)=0.0625，需要放大到合理范围
        # 使用lr_mul=16可以达到标准Transformer(d_model=512)的学习率
        lr_mul = 16.0 if hp_cont.d_model < 512 else 1.0
        optimizer = ScheduledOptim(opt_inst, hp_cont.d_model, hp_cont.n_warmup_steps, lr_mul=lr_mul)
        scheduler = None  # don't change this
    else:
        optimizer = opt_inst
        scheduler = MultiStepLR(optimizer, milestones=[8*i for i in range(1, hp_cont.epoch//8 + 1)], gamma=0.1)
#===================================================================================================================================
        
    # --- save history for the plots ---
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_bleu": [], "lr": [], "epoch_time": []}
    
    # --------------- resume (epoch-level) ---------------
    start_epoch, hist = ckpt.resume(model, sched_optim=optimizer, scheduler=scheduler)
    
    reg_tag = reg_name
    drop_tag = "dropout" if hp_cont.dropout != 0.0 else "no_dropout"
    norm_pos_tag = "pre" if norm_inside_residual else "post"
    norm_tag = norm_name
    warm_up_tag = "warmup" if warm_up_bool else "no_warmup"
    
    # --- train loop ---
    for epoch_i in range(start_epoch, hp_cont.epoch):
        start_time = time.time()
        tr_loss, tr_acc = train_epoch(model, training_data, optimizer, hp_cont, device, warm_up_bool=warm_up_bool)
        va_loss, va_bleu = eval_epoch(model, validation_data, device, hp_cont)
        if not warm_up_bool and scheduler is not None:
            scheduler.step()
        if warm_up_bool:
            cur_lr = optimizer._optimizer.param_groups[0]['lr']
        else:
            cur_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_bleu"].append(va_bleu)
        hist["lr"].append(cur_lr)
        hist["epoch_time"].append(elapsed)
        print(
            f"[{drop_tag}+{reg_tag}] "
            f"[{norm_pos_tag}+{norm_tag}] "
            f"[{warm_up_tag}] lr={cur_lr} |"
            f"Epoch {epoch_i+1}/{hp_cont.epoch} | "
            f"Train: loss={tr_loss:.4f}, acc={tr_acc*100:.2f}% | "
            f"Val: loss={va_loss:.4f}, BLEU={va_bleu:.2f} | "
            f"time: {elapsed:.2f}s"
        )
        
        # ---- save one checkpoint per completed epoch (highly recommended) ----
        ckpt.save(
            epoch=epoch_i + 1,
            model=model,
            sched_optim=optimizer,
            scheduler=scheduler,
            hist=hist,
            hp_cont=hp_cont
        )
    
    # ------- Final Validation (test set has no labels) -------
    print(
        f"[{drop_tag}+{reg_tag}] "
        f"[{norm_pos_tag}+{norm_tag}] "
        f"[{warm_up_tag}] "
        f"Training completed | Final Validation BLEU: {hist['val_bleu'][-1]:.2f}"
    )
    
    # Save final checkpoint
    ckpt.save(
        epoch=hp_cont.epoch,
        model=model,
        sched_optim=optimizer,
        scheduler=scheduler,
        hist=hist,
        hp_cont=hp_cont
    )

if __name__ == '__main__':
    import sys
    
    # Add RAdam to path
    radam_path = os.path.abspath('./RAdam')
    if radam_path not in sys.path:
        sys.path.insert(0, radam_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Hyperparameters
    from types import SimpleNamespace
    hp_cont = SimpleNamespace()
    
    hp_cont.data_pkl = 'dataset/akkadian.pkl'  # pkl文件
    hp_cont.epoch = 50
    hp_cont.batch_size = 256
    hp_cont.max_token_seq_len = 100  # 句子最大长度
    hp_cont.label_smoothing = True
    hp_cont.d_model = 256
    hp_cont.d_word_vec = 256
    hp_cont.d_inner_hid = 1024
    hp_cont.d_k = 64
    hp_cont.d_v = 64
    hp_cont.n_head = 4
    hp_cont.n_layers = 6
    hp_cont.betas = (0.9, 0.98)
    hp_cont.lr = 1e-3
    hp_cont.wd = 1e-2
    hp_cont.embs_share_weight = True
    hp_cont.proj_share_weight = True
    hp_cont.scale_emb_or_prj = 'emb'
    hp_cont.seed = 42
    
    save_dir = "checkpoints_Transformer"
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("Training Best Configuration: Dropout + Pre-LN + Warmup")
    print("="*80)
    
    # 最优配置：dropout + pre-LN + warmup (BLEU=17.54)
    ckpt = Checkpointer(os.path.join(save_dir, "best_model.pth"), device=device)
    main(ckpt, device, hp_cont, dropout_bool=True, reg_name="no_reg", 
         norm_name="layernorm", norm_inside_residual=True, warm_up_bool=True)
    
    # ========== 其他实验配置（已注释） ==========
    # # Exp 1: dropout + post-LN + warmup (BLEU=6.94)
    # print("\n[1/10] dropout + post-LN + warmup")
    # ckpt = Checkpointer(os.path.join(save_dir, "dropout_post_layernorm_warmup_no_reg.pth"), device=device)
    # main(ckpt, device, hp_cont, dropout_bool=True, reg_name="no_reg", 
    #      norm_name="layernorm", norm_inside_residual=False, warm_up_bool=True)
    
    # # Exp 2: no dropout + post-LN + warmup (BLEU=13.96)
    # print("\n[2/10] no dropout + post-LN + warmup")
    # ckpt = Checkpointer(os.path.join(save_dir, "no_dropout_post_layernorm_warmup_no_reg.pth"), device=device)
    # main(ckpt, device, hp_cont, dropout_bool=False, reg_name="no_reg", 
    #      norm_name="layernorm", norm_inside_residual=False, warm_up_bool=True)
    
    # # Exp 8: L2 regularization (BLEU=10.61)
    # print("\n[8/10] dropout + pre-LN + no warmup + L2")
    # ckpt = Checkpointer(os.path.join(save_dir, "dropout_pre_layernorm_no_warmup_l2.pth"), device=device)
    # main(ckpt, device, hp_cont, dropout_bool=True, reg_name="l2", 
    #      norm_name="layernorm", norm_inside_residual=True, warm_up_bool=False)
    
    # # Exp 9: weight decay (BLEU=10.61)
    # print("\n[9/10] dropout + pre-LN + no warmup + weight_decay")
    # ckpt = Checkpointer(os.path.join(save_dir, "dropout_pre_layernorm_no_warmup_weight_decay.pth"), device=device)
    # main(ckpt, device, hp_cont, dropout_bool=True, reg_name="weight_decay", 
    #      norm_name="layernorm", norm_inside_residual=True, warm_up_bool=False)
    
    
    print("\n" + "="*80)
    print("✓ Training completed!")
    print("="*80)
