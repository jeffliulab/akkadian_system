import os, torch, random, numpy as np
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class Checkpointer:
    path: str
    device: torch.device
        
    def _model_state(self, model):
        return (model.module.state_dict()
                if hasattr(model, "module") else model.state_dict())

    def _current_lr(self, optim_or_wrap):
        try:
            return optim_or_wrap._optimizer.param_groups[0]["lr"]
        except Exception:
            try:
                return optim_or_wrap.param_groups[0]["lr"]
            except Exception:
                return None

    def save(self, epoch: int, model, sched_optim=None, scheduler=None, hist=None, hp_cont=None, extra: dict=None):
        """
        Atomically save a checkpoint to `path`.
        This reduces the risk of a half-written file if the job is killed mid-save.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": self._model_state(model),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if sched_optim is not None:
            inner = getattr(sched_optim, "_optimizer", None)
            if inner is not None:
                payload["inner_optimizer_state_dict"] = inner.state_dict()
                payload["sched_wrapper"] = {
                    "class": type(sched_optim).__name__,
                    "n_steps": getattr(sched_optim, "n_steps", 0),
                    "d_model": getattr(sched_optim, "d_model", None),
                    "n_warmup_steps": getattr(sched_optim, "n_warmup_steps", None),
                    "lr_mul": getattr(sched_optim, "lr_mul", 1.0),
                }
            else:
                payload["optimizer_state_dict"] = sched_optim.state_dict()

            payload["current_lr"] = self._current_lr(sched_optim)
            
        if scheduler is not None:
            try:
                payload["scheduler_state_dict"] = scheduler.state_dict()
                payload["scheduler_class"] = type(scheduler).__name__
            except Exception as e:
                print(f"[save] Warning: failed to save scheduler state: {e}")

        if hist is not None:
            payload["hist"] = deepcopy(hist)

        if hp_cont is not None:
            try:
                payload["hp_cont"] = deepcopy(vars(hp_cont))
            except Exception:
                payload["hp_cont"] = None

        if extra is not None:
            payload["extra"] = deepcopy(extra)

        tmp = self.path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.path)


    def resume(self, model, sched_optim=None, scheduler=None, hist=None):
        """
        Resume training state from `ckpt_path` if it exists.

        Restores:
          - model parameters
          - optimizer state (if provided)
          - RNG states (CPU + all CUDA devices, if available)
          - last finished epoch
          - training/validation history

        Returns:
          start_epoch (int): last finished epoch number (0 if none)
          hist      (dict): metric history dict with lists
        """
        p = Path(self.path)
        start_epoch = 0
        hist = hist or {"train_loss": [], "train_acc": [], "val_loss": [], "val_bleu": [], "lr": [], "epoch_time": []}
        if not p.exists():
            print(f"[resume] No checkpoint at {p}; starting from scratch.")
            print("-------------------------------------------------------------------")
            return start_epoch, hist

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if sched_optim is not None:
            inner = getattr(sched_optim, "_optimizer", None)
            if inner is not None and "inner_optimizer_state_dict" in ckpt:
                inner.load_state_dict(ckpt["inner_optimizer_state_dict"])
                sw = ckpt.get("sched_wrapper", {})
                sched_optim.n_steps        = sw.get("n_steps", getattr(sched_optim, "n_steps", 0))
                sched_optim.d_model        = sw.get("d_model", getattr(sched_optim, "d_model", None))
                sched_optim.n_warmup_steps = sw.get("n_warmup_steps", getattr(sched_optim, "n_warmup_steps", None))
                if "lr_mul" in sw:
                    setattr(sched_optim, "lr_mul", sw["lr_mul"])
            elif "optimizer_state_dict" in ckpt:
                try:
                    sched_optim.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"[resume] Warning: failed to load optimizer state: {e}")
                    
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"[resume] Warning: failed to load scheduler state: {e}")

        rng = ckpt.get("rng", {})
        try:
            if "python" in rng: random.setstate(rng["python"])
            if "numpy"  in rng: np.random.set_state(rng["numpy"])
            if "torch"  in rng: torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda_all", None) is not None:
                torch.cuda.set_rng_state_all(rng["cuda_all"])
        except Exception:
            pass

        start_epoch = ckpt.get("epoch", 0)
        if "hist" in ckpt and ckpt["hist"] is not None:
            hist = ckpt["hist"]
        print(f"[resume] Loaded {p}: last finished epoch = {start_epoch}")
        print("-------------------------------------------------------------------")
        return start_epoch, hist

    def remove(self):
        """
        Delete checkpoint file if it exists (fresh start).
        """
        p = Path(self.path)
        if p.exists():
            p.unlink()
            print(f"Removed checkpoint: {p.resolve()}")
        else:
            print(f"No checkpoint found at: {p.resolve()}")

