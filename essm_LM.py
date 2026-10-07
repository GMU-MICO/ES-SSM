import argparse
import gc
import json
import math
import os
import random
import warnings
from contextlib import contextmanager
from pathlib import Path
from runtime_utils import atomic_save, build_eval_cache, parquet_files as _parquet_files, hankel_basis

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


CFG = {
    "seed": 86,
    "vocab_size": 258,
    "seq_len": 2048,
    "d_model": 1262,
    "n_layers": 20,
    "K_max": 32,
    "batch_size": 16,
    "grad_accum_steps": 2,
    "grad_ckpt": False,
    "gate_bias_init": -2.0,
    "dropout": 0.1,
    "stoch_depth": 0.1,
    "lr": 3e-4,
    "warmup_ratio": 0.01,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "target_tokens": 30_000_000_000,
    "tokens_between_val": 1_000_000_000,
    "budget_warmup_tokens": 1_000_000_000,
    "budget_k_min": 2,
    "budget_full_every": 8,
    "budget_ks": [2, 3, 4, 5, 6, 8, 12, 16, 24, 32],
    "eval_k": [2, 3, 4, 5, 6, 8, 12, 16, 24, 32],
    "K_chunk": 8,
    "eval_batches": 200,
    "test_batches": 250,
    "filter_iters": 60,
    "filter_oversample": 16,
    "filter_tol": 1e-6,
    "num_workers": 4,
    "val_segments": 1000,
    "test_segments": 1000,
}


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_DIST = WORLD_SIZE > 1
IS_MASTER = RANK == 0


if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not IS_DIST:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16


AMP_DTYPE = _amp_dtype()


@contextmanager
def autocast():
    if torch.cuda.is_available():
        with torch.amp.autocast(device_type=AMP_DEVICE, dtype=AMP_DTYPE):
            yield
    else:
        yield


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def emit(msg):
    if IS_MASTER:
        print(msg, flush=True)


def _b2t(b):
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).to(torch.int64).add_(2)


class StreamingByteDataset(IterableDataset):
    def __init__(self, data_dir, split, seed, distributed=True):
        if pq is None:
            raise ImportError("Local parquet loading requires pyarrow.")
        self.data_dir = data_dir
        self.split = split
        self.seed = seed
        self.seg = CFG["seq_len"] + 1
        self.files = _parquet_files(data_dir, split)
        self.distributed = distributed

    def __iter__(self):
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        nw = 1 if wi is None else wi.num_workers
        cid = RANK * nw + wid if self.distributed else wid
        ctot = WORLD_SIZE * nw if self.distributed else nw
        if len(self.files) < ctot:
            raise ValueError(f"Need at least {ctot} parquet shards for the selected ranks/workers; found {len(self.files)}. Reduce num_workers or ranks.")
        shard = self.files[cid::ctot] if ctot > 1 else self.files
        if not shard:
            return
        rng = random.Random(self.seed + 13 * cid)
        buf = bytearray()
        while True:
            order = list(shard)
            rng.shuffle(order)
            nonempty = False
            for fp in order:
                pf = pq.ParquetFile(fp)
                for rb in pf.iter_batches(batch_size=256, columns=["text"]):
                    for t in rb.column(0).to_pylist():
                        if not t:
                            continue
                        nonempty = True
                        buf.extend(t.encode("utf-8", errors="ignore"))
                        buf.extend(b"\n")
                        while len(buf) >= self.seg:
                            yield _b2t(bytes(buf[: self.seg]))
                            del buf[: self.seg - 1]
            if not nonempty:
                raise ValueError(f"No nonempty text in {self.split} shards.")


class CachedSegments(Dataset):
    def __init__(self, t):
        self.t = t

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, i):
        return self.t[i].to(torch.long)


@torch.no_grad()
def build_cache(data_dir, output_dir, split, n, seed):
    return build_eval_cache(data_dir, Path(output_dir) / "eval_cache", split, n, seed, CFG["seq_len"], RANK, WORLD_SIZE, IS_DIST)


@torch.no_grad()
def hankel_top_k(cache_dir, L, K):
    return hankel_basis(cache_dir, L, K, CFG["filter_iters"], CFG["filter_oversample"], CFG["filter_tol"])


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < (1 - self.p)
        return x * mask / (1 - self.p)


class ESSMLayer(nn.Module):
    def __init__(self, cache_dir, d, L, K, dp=0.0):
        super().__init__()
        self.K, self.L = K, L
        sigma, phi = hankel_top_k(cache_dir, L, K)
        self.register_buffer("sigma", sigma)
        with torch.no_grad():
            phi_f = torch.fft.rfft(phi.T.float(), n=2 * L, dim=-1)
        self.register_buffer("phi_f_ri", torch.view_as_real(phi_f).contiguous())
        self.M = nn.Parameter(torch.randn(K, d, d) * 1e-3)
        self.gate = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, K))
        nn.init.constant_(self.gate[-1].bias, float(CFG["gate_bias_init"]))
        self.dp = DropPath(dp)

    def forward(self, u, K_rt=None):
        B, L, D = u.shape
        n = 2 * L
        K = self.K if K_rt is None else max(CFG["budget_k_min"], min(self.K, int(K_rt)))
        w = (torch.sigmoid(self.gate(u)[:, :, :K].float()) / math.sqrt(K)).to(u.dtype)
        chunk = int(CFG.get("K_chunk", 0))
        chunked = (not self.training) and (0 < chunk < K)
        if not chunked:
            with torch.amp.autocast(device_type=AMP_DEVICE, enabled=False):
                xf = torch.fft.rfft(u.float().permute(0, 2, 1), n=n, dim=-1)
                pf = torch.view_as_complex(self.phi_f_ri[:K].contiguous())
                Uo = torch.fft.irfft(xf.unsqueeze(2) * pf.unsqueeze(0).unsqueeze(0), n=n, dim=-1)[..., :L]
                Uo = Uo.permute(0, 3, 2, 1)
                if Uo.dtype != u.dtype:
                    Uo = Uo.to(dtype=u.dtype, memory_format=torch.contiguous_format)
            y = torch.einsum("blkd,kdo->blo", Uo * w.unsqueeze(-1), self.M[:K].to(u.dtype))
        else:
            with torch.amp.autocast(device_type=AMP_DEVICE, enabled=False):
                xf = torch.fft.rfft(u.float().permute(0, 2, 1), n=n, dim=-1)
                pf_full = torch.view_as_complex(self.phi_f_ri[:K].contiguous())
            y = torch.zeros(B, L, D, device=u.device, dtype=u.dtype)
            for s in range(0, K, chunk):
                e = min(s + chunk, K)
                with torch.amp.autocast(device_type=AMP_DEVICE, enabled=False):
                    pf_c = pf_full[s:e]
                    Uo_c = torch.fft.irfft(xf.unsqueeze(2) * pf_c.unsqueeze(0).unsqueeze(0), n=n, dim=-1)[..., :L]
                    Uo_c = Uo_c.permute(0, 3, 2, 1)
                    if Uo_c.dtype != u.dtype:
                        Uo_c = Uo_c.to(dtype=u.dtype, memory_format=torch.contiguous_format)
                y = y + torch.einsum("blkd,kdo->blo", Uo_c * w[:, :, s:e].unsqueeze(-1), self.M[s:e].to(u.dtype))
                del Uo_c
        return self.dp(y), K


class ESSSMLanguageModel(nn.Module):
    def __init__(self, cache_dir):
        super().__init__()
        d = CFG["d_model"]
        L = CFG["seq_len"]
        K = CFG["K_max"]
        nl = CFG["n_layers"]
        self.K_max = K
        self.d_model = d
        self.use_ckpt = CFG["grad_ckpt"]
        self.emb = nn.Embedding(CFG["vocab_size"], d, padding_idx=0)
        self.drop = nn.Dropout(CFG["dropout"])
        sd = torch.linspace(0, CFG["stoch_depth"], nl).tolist()
        self.layers = nn.ModuleList([ESSMLayer(cache_dir, d, L, K, sd[i]) for i in range(nl)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d * 4),
                nn.GELU(),
                nn.Dropout(CFG["dropout"]),
                nn.Linear(d * 4, d),
            ) for _ in range(nl)
        ])
        self.ffn_dp = nn.ModuleList([DropPath(sd[i]) for i in range(nl)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, CFG["vocab_size"], bias=False)
        nn.init.normal_(self.emb.weight, 0, 0.02)
        self.head.weight = self.emb.weight

    def _block(self, x, i, K_rt):
        y, _ = self.layers[i](self.n1[i](x), K_rt=K_rt)
        x = x + y
        x = x + self.ffn_dp[i](self.ffn[i](self.n2[i](x)))
        return x

    def forward(self, tok, K_rt=None):
        x = self.drop(self.emb(tok))
        for i in range(len(self.layers)):
            if self.use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(self._block, x, i, K_rt, use_reentrant=False)
            else:
                x = self._block(x, i, K_rt)
        return self.head(self.ln_f(x))


def build_compact(cache_dir, source, K):
    old = CFG["K_max"]
    try:
        CFG["K_max"] = int(K)
        compact = ESSSMLanguageModel(cache_dir)
    finally:
        CFG["K_max"] = old
    with torch.no_grad():
        compact.emb.weight.copy_(source.emb.weight.detach().to(compact.emb.weight.device))
        compact.ln_f.load_state_dict(source.ln_f.state_dict())
        for d, s in zip(compact.n1, source.n1):
            d.load_state_dict(s.state_dict())
        for d, s in zip(compact.n2, source.n2):
            d.load_state_dict(s.state_dict())
        for d, s in zip(compact.ffn, source.ffn):
            d.load_state_dict(s.state_dict())
        for dl, sl in zip(compact.layers, source.layers):
            dl.sigma.copy_(sl.sigma[:K].detach().to(dl.sigma.device))
            dl.phi_f_ri.copy_(sl.phi_f_ri[:K].detach().to(dl.phi_f_ri.device))
            dl.M.copy_(sl.M[:K].detach().to(dl.M.device))
            dl.gate[0].load_state_dict(sl.gate[0].state_dict())
            dl.gate[2].weight.copy_(sl.gate[2].weight[:K].detach().to(dl.gate[2].weight.device))
            dl.gate[2].bias.copy_(sl.gate[2].bias[:K].detach().to(dl.gate[2].bias.device))
    compact.head.weight = compact.emb.weight
    compact.use_ckpt = False
    return compact


class BudgetSampler:
    def __init__(self):
        self.rng = random.Random(CFG["seed"] + 777)
        self.ks = sorted(CFG["budget_ks"])
        self._eligible = [k for k in self.ks if k >= CFG["budget_k_min"]]
        self._k_max = CFG["K_max"]
        self._lo = max(CFG["budget_k_min"], min(self.ks))
        self._hi = max(self.ks)
        self._log_ratio = math.log(self._hi / self._lo)

    def state_dict(self):
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if state and "rng" in state:
            self.rng.setstate(state["rng"])

    def _draw(self, step, tokens_seen):
        if tokens_seen < CFG["budget_warmup_tokens"]:
            return self._k_max
        fe = int(CFG["budget_full_every"])
        if fe > 0 and step % fe == 0:
            return self._k_max
        u = self.rng.random()
        target = self._lo * math.exp(u * self._log_ratio)
        return min(self._eligible, key=lambda k: abs(k - target))

    def sample(self, step, tokens_seen):
        if IS_DIST:
            if IS_MASTER:
                k = self._draw(step, tokens_seen)
                buf = torch.tensor([k], dtype=torch.long, device=DEVICE)
            else:
                buf = torch.empty(1, dtype=torch.long, device=DEVICE)
            dist.broadcast(buf, src=0)
            return int(buf.item())
        return self._draw(step, tokens_seen)


def build_optimizer(model):
    decay, no_decay, seen = [], [], set()
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 1 or any(k in n.lower() for k in ("bias", "norm", "emb", "ln")):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": CFG["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=CFG["lr"], betas=(0.9, 0.95), eps=1e-8,
    )


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler(AMP_DEVICE, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


class CompiledInference(nn.Module):
    def __init__(self, model, enabled=False):
        super().__init__()
        self.model = model.eval()
        self.compiled = None
        self.checked_inputs = set()
        self.execution = "eager"
        if enabled:
            try:
                device = next(model.parameters()).device
                mode = "reduce-overhead" if device.type == "cuda" else "default"
                self.compiled = torch.compile(model, mode=mode, dynamic=False, fullgraph=False)
                self.execution = "pending"
            except Exception as exc:
                self._fallback(exc)
        self.eval()

    def _fallback(self, exc):
        self.compiled = None
        self.execution = "fallback"
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        warnings.warn(f"Compiled inference fell back to eager execution: {detail}", RuntimeWarning)

    def forward(self, tokens):
        if self.training:
            raise RuntimeError("CompiledInference only supports evaluation.")
        if self.compiled is None:
            return self.model(tokens)
        try:
            compiler = getattr(torch, "compiler", None)
            if tokens.is_cuda and hasattr(compiler, "cudagraph_mark_step_begin"):
                compiler.cudagraph_mark_step_begin()
            output = self.compiled(tokens)
            key = (tuple(tokens.shape), tuple(tokens.stride()), tokens.device, tokens.dtype)
            if key not in self.checked_inputs:
                reference = self.model(tokens)
                if output.shape != reference.shape or output.dtype != reference.dtype:
                    raise RuntimeError("Compiled output shape or dtype differs from eager output.")
                if not torch.isfinite(output).all() or not torch.isfinite(reference).all():
                    raise RuntimeError("Non-finite output during compiled-inference validation.")
                error = (output.float() - reference.float()).norm()
                error = float(error / reference.float().norm().clamp_min(1e-12))
                tolerance = 1e-4 if reference.dtype == torch.float32 else 0.005
                if not math.isfinite(error) or error > tolerance:
                    raise RuntimeError(f"Compiled output relative L2 error {error:.6g} exceeds {tolerance}.")
                self.checked_inputs.add(key)
            self.execution = "compiled"
            return output
        except Exception as exc:
            self._fallback(exc)
            return self.model(tokens)


def checkpoint_selection(mode="full", budgets=None, weights=None, full_bpb_ceiling=None):
    if mode not in ("full", "multi"):
        raise ValueError("Checkpoint selection must be 'full' or 'multi'.")
    if mode == "full":
        if budgets is not None or weights is not None or full_bpb_ceiling is not None:
            raise ValueError("Budget weights and the full-budget ceiling require --checkpoint-selection multi.")
        return {"mode": "full", "budgets": [CFG["K_max"]], "weights": [1.0], "full_bpb_ceiling": None}
    ks = [int(k) for k in budgets.split(",")] if budgets is not None else list(CFG["eval_k"])
    ws = [float(w) for w in weights.split(",")] if weights is not None else [1.0] * len(ks)
    if not ks or len(ks) != len(set(ks)) or any(k < CFG["budget_k_min"] or k > CFG["K_max"] for k in ks):
        raise ValueError("Selection budgets must be unique supported integer budgets.")
    if len(ws) != len(ks) or any(not math.isfinite(w) or w <= 0 for w in ws) or not math.isfinite(sum(ws)):
        raise ValueError("Selection weights must be finite, positive, and match the budget count.")
    if full_bpb_ceiling is not None and (not math.isfinite(full_bpb_ceiling) or full_bpb_ceiling <= 0):
        raise ValueError("The full-budget BPB ceiling must be finite and positive.")
    return {"mode": "multi", "budgets": ks, "weights": ws, "full_bpb_ceiling": full_bpb_ceiling}


def checkpoint_score(metrics, selection):
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite validation BPB cannot select a checkpoint.")
    total = sum(selection["weights"])
    return sum(metrics[k] * (w / total) for k, w in zip(selection["budgets"], selection["weights"]))


@torch.no_grad()
def eval_bpb(model, loader, K=None, elastic=True, max_batches=None, distributed=True):
    model.eval()
    nll, tok = 0.0, 0
    max_batches = max_batches if max_batches else CFG["eval_batches"]
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = b.to(DEVICE, non_blocking=True)
        x, y = b[:, :-1], b[:, 1:]
        with autocast():
            lo = model(x, K_rt=K) if elastic else model(x)
            loss = F.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1))
        nll += loss.item() * y.numel()
        tok += y.numel()
    if IS_DIST and distributed:
        stats = torch.tensor([nll, float(tok)], device=DEVICE)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        nll, tok = stats[0].item(), stats[1].item()
    if tok <= 0:
        raise ValueError("Evaluation produced no target tokens.")
    value = (nll / tok) / math.log(2)
    if not math.isfinite(value):
        raise FloatingPointError("Evaluation BPB is non-finite.")
    return value


def train_loop(model, train_dl, val_dl, ckpt_dir, selection=None):
    selection = checkpoint_selection() if selection is None else selection
    model.to(DEVICE)
    if hasattr(model, "use_ckpt"):
        model.use_ckpt = CFG["grad_ckpt"]
    core = model.module if IS_DIST and hasattr(model, "module") else model

    target_tokens = CFG["target_tokens"]
    tokens_per_step = CFG["batch_size"] * CFG["grad_accum_steps"] * CFG["seq_len"] * WORLD_SIZE
    total_steps = max(1, target_tokens // tokens_per_step)
    wu = max(1, int(total_steps * CFG["warmup_ratio"]))

    opt = build_optimizer(model)

    def lrf(s):
        if s < wu:
            return (s + 1) / wu
        p = (s - wu) / max(1, total_steps - wu)
        return CFG["min_lr_ratio"] + (1 - CFG["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrf)
    use_scaler = AMP_DTYPE == torch.float16
    scaler = make_scaler(use_scaler)
    bud = BudgetSampler()

    it = iter(train_dl)
    g = 0
    tokens_seen = 0
    next_val = CFG["tokens_between_val"]
    best_bpb = float("inf")
    best_score = float("inf")
    selected_path = None

    while tokens_seen < target_tokens:
        model.train()
        K_cur = bud.sample(g, tokens_seen)
        opt.zero_grad(set_to_none=True)
        step_tokens = 0
        for _ in range(CFG["grad_accum_steps"]):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_dl)
                batch = next(it)
            batch = batch.to(DEVICE, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            with autocast():
                logits = model(x, K_rt=K_cur)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            scaled = loss / CFG["grad_accum_steps"]
            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            step_tokens += y.numel()
        if use_scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        if use_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        g += 1
        tokens_seen += step_tokens * WORLD_SIZE

        if tokens_seen >= next_val or tokens_seen >= target_tokens:
            next_val += CFG["tokens_between_val"]
            ks = list(dict.fromkeys([CFG["K_max"]] + selection["budgets"]))
            metrics = {
                k: eval_bpb(model, val_dl, K=k, elastic=True, max_batches=CFG["eval_batches"])
                for k in ks
            }
            vbpb = metrics[CFG["K_max"]]
            score = checkpoint_score(metrics, selection)
            ceiling = selection["full_bpb_ceiling"]
            eligible = ceiling is None or vbpb <= ceiling
            emit(json.dumps({"step": g, "tokens_seen": tokens_seen, "validation_bpb": metrics,
                             "selection_score": score, "eligible": eligible}, sort_keys=True))
            metadata = {
                "cfg": dict(CFG),
                "step": g,
                "tokens_seen": tokens_seen,
                "validation_bpb": metrics,
                "selection_protocol": selection,
            }
            if vbpb < best_bpb - 1e-4:
                best_bpb = vbpb
                if selection["mode"] == "full" and ckpt_dir is not None:
                    selected_path = ckpt_dir / "best.pt"
                if IS_MASTER and ckpt_dir is not None:
                    atomic_save({"model": core.state_dict(), **metadata, "selected_by": "full",
                                "selection_score": vbpb}, str(ckpt_dir / "best.pt"))
            if selection["mode"] == "multi" and eligible and score < best_score - 1e-4:
                best_score = score
                if ckpt_dir is not None:
                    selected_path = ckpt_dir / "best_multibudget.pt"
                    if IS_MASTER:
                        atomic_save({"model": core.state_dict(), **metadata, "selected_by": "multi",
                                    "selection_score": score}, str(selected_path))
    if IS_DIST:
        dist.barrier()
    if selection["mode"] == "multi" and not math.isfinite(best_score):
        raise RuntimeError("No checkpoint satisfied the prescribed full-budget BPB ceiling. The full-budget best is saved as best.pt.")
    return selected_path


def maybe_init_dist():
    global IS_DIST
    if DEVICE.type == "cuda":
        torch.cuda.set_device(DEVICE)
    if WORLD_SIZE > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=RANK, world_size=WORLD_SIZE)
        IS_DIST = True


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data-dir", type=str, required=True)
    pa.add_argument("--output-dir", type=str, required=True)
    pa.add_argument("--filter-cache-dir", type=str, default="./cache/filter")
    pa.add_argument("--inference-only", action="store_true")
    pa.add_argument("--checkpoint", type=str, default=None)
    pa.add_argument("--compile-inference", action="store_true")
    pa.add_argument("--checkpoint-selection", choices=("full", "multi"), default="full")
    pa.add_argument("--selection-budgets", type=str)
    pa.add_argument("--selection-weights", type=str)
    pa.add_argument("--full-bpb-ceiling", type=float)
    pa.add_argument("--num-workers", type=int, default=None)
    pa.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = pa.parse_args()
    if args.checkpoint and not args.inference_only:
        pa.error("--checkpoint is for --inference-only; training resume is not implemented by this entrypoint.")
    if args.num_workers is not None:
        if args.num_workers < 0:
            pa.error("--num-workers must be nonnegative")
        CFG["num_workers"] = args.num_workers
    try:
        selection = checkpoint_selection(args.checkpoint_selection, args.selection_budgets,
                                         args.selection_weights, args.full_bpb_ceiling)
    except ValueError as exc:
        pa.error(str(exc))

    maybe_init_dist()
    set_seed(CFG["seed"] + RANK)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.filter_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    val_t = build_cache(args.data_dir, args.output_dir, "validation", CFG["val_segments"], CFG["seed"] + 111)
    val_dl = DataLoader(CachedSegments(val_t), batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    model = ESSSMLanguageModel(args.filter_cache_dir)
    if IS_DIST:
        model = model.to(DEVICE)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK] if DEVICE.type == "cuda" else None,
            output_device=LOCAL_RANK if DEVICE.type == "cuda" else None,
            find_unused_parameters=False,
        )

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    selected_path = ckpt_dir / ("best_multibudget.pt" if selection["mode"] == "multi" else "best.pt")
    if not args.inference_only:
        train_ds = StreamingByteDataset(args.data_dir, "train", CFG["seed"])
        train_dl = DataLoader(
            train_ds,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_workers"],
            pin_memory=True,
            collate_fn=torch.stack,
            drop_last=True,
            persistent_workers=CFG["num_workers"] > 0,
        )
        selected_path = train_loop(model, train_dl, val_dl, ckpt_dir, selection=selection)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else selected_path
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ck = torch.load(str(checkpoint_path), map_location="cpu")
    (model.module if IS_DIST else model).load_state_dict(ck["model"])
    del ck
    test_t = build_cache(args.data_dir, args.output_dir, "test", CFG["test_segments"], CFG["seed"] + 222)
    test_dl = DataLoader(CachedSegments(test_t), batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    core = model.module if IS_DIST else model
    core.use_ckpt = False

    if IS_MASTER:
        core = core.cpu()
        cleanup()
        for K in CFG["eval_k"]:
            compact = build_compact(args.filter_cache_dir, core, K)
            compact = compact.to(DEVICE)
            compiler = getattr(torch, "compiler", None)
            if args.compile_inference and hasattr(compiler, "reset"):
                compiler.reset()
            inference = CompiledInference(compact, enabled=args.compile_inference)
            v = eval_bpb(inference, val_dl, elastic=False, max_batches=CFG["eval_batches"], distributed=False)
            t = eval_bpb(inference, test_dl, elastic=False, max_batches=CFG["test_batches"], distributed=False)
            emit(f"K={K} val={v:.4f} test={t:.4f} execution={inference.execution}")
            del inference, compact
            cleanup()

    if IS_DIST:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
