import argparse
import gc
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path
from runtime_utils import atomic_save, build_eval_cache, parquet_files as _parquet_files

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


SIZES = [
    {"K": 2,  "d_model": 1024, "n_layers": 24, "n_heads": 8,  "head_kv": 128, "head_v": 128, "d_ffn": 2816},
    {"K": 8,  "d_model": 1280, "n_layers": 24, "n_heads": 10, "head_kv": 128, "head_v": 128, "d_ffn": 3584},
    {"K": 16, "d_model": 1536, "n_layers": 24, "n_heads": 12, "head_kv": 128, "head_v": 128, "d_ffn": 4480},
    {"K": 32, "d_model": 2048, "n_layers": 24, "n_heads": 16, "head_kv": 128, "head_v": 128, "d_ffn": 5376},
]


CFG = {
    "seed": 86,
    "vocab_size": 258,
    "seq_len": 2048,
    "batch_size": 16,
    "grad_accum_steps": 2,
    "grad_ckpt": False,
    "rms_norm_eps": 1e-5,
    "chunk_size": 64,
    "lr": 3e-4,
    "warmup_ratio": 0.01,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "target_tokens": 30_000_000_000,
    "tokens_between_val": 1_000_000_000,
    "eval_batches": 200,
    "test_batches": 250,
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


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return (self.weight.float() * x).to(dt)


def gla_chunkwise(q, k, v, gk, chunk_size):
    B, H, L, Dk = q.shape
    Dv = v.size(-1)
    Cs = chunk_size
    assert L % Cs == 0
    nc = L // Cs
    q = q.view(B, H, nc, Cs, Dk)
    k = k.view(B, H, nc, Cs, Dk)
    v = v.view(B, H, nc, Cs, Dv)
    gk = gk.view(B, H, nc, Cs, Dk)
    state = q.new_zeros(B, H, Dk, Dv)
    out_chunks = []
    log_g_cum = gk.cumsum(dim=3)
    for c in range(nc):
        gc = log_g_cum[:, :, c]
        lg_end = gc[:, :, -1:, :]
        attn = torch.einsum("bhld,bhmd->bhlm", q[:, :, c] * gc.exp(), k[:, :, c] * (-gc).exp())
        mask = torch.tril(torch.ones(Cs, Cs, device=q.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, 0.0)
        intra = torch.einsum("bhlm,bhmd->bhld", attn, v[:, :, c])
        inter = torch.einsum("bhlk,bhkd->bhld", q[:, :, c] * gc.exp(), state)
        out_chunks.append(intra + inter)
        k_decay = k[:, :, c] * (lg_end - gc).exp()
        state = state * lg_end[:, :, 0, :, None].exp() + torch.einsum("bhmd,bhme->bhde", k_decay, v[:, :, c])
    out = torch.cat(out_chunks, dim=2)
    return out


class GLAttention(nn.Module):
    def __init__(self, d, n_heads, head_kv, head_v):
        super().__init__()
        self.n_heads = n_heads
        self.head_kv = head_kv
        self.head_v = head_v
        self.q = nn.Linear(d, n_heads * head_kv, bias=False)
        self.k = nn.Linear(d, n_heads * head_kv, bias=False)
        self.v = nn.Linear(d, n_heads * head_v, bias=False)
        self.gk_proj = nn.Sequential(nn.Linear(d, 16, bias=False), nn.Linear(16, n_heads * head_kv, bias=True))
        self.o = nn.Linear(n_heads * head_v, d, bias=False)
        self.gate = nn.Linear(d, n_heads * head_v, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        Dk = self.head_kv
        Dv = self.head_v
        q = self.q(x).view(B, L, H, Dk).transpose(1, 2)
        k = F.silu(self.k(x)).view(B, L, H, Dk).transpose(1, 2)
        v = self.v(x).view(B, L, H, Dv).transpose(1, 2)
        gk = F.logsigmoid(self.gk_proj(x)).view(B, L, H, Dk).transpose(1, 2) / 16.0
        cs = CFG["chunk_size"]
        if L % cs != 0:
            cs = L
        out = gla_chunkwise(q, k, v, gk, cs)
        out = out.transpose(1, 2).contiguous().view(B, L, H * Dv)
        out = out * F.silu(self.gate(x))
        return self.o(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d, d_ffn):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ffn, bias=False)
        self.w_up = nn.Linear(d, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GLABlock(nn.Module):
    def __init__(self, d, n_heads, head_kv, head_v, d_ffn, eps):
        super().__init__()
        self.norm1 = RMSNorm(d, eps)
        self.attn = GLAttention(d, n_heads, head_kv, head_v)
        self.norm2 = RMSNorm(d, eps)
        self.ffn = SwiGLUFFN(d, d_ffn)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GLALM(nn.Module):
    def __init__(self, sz):
        super().__init__()
        d = sz["d_model"]
        n = sz["n_layers"]
        h = sz["n_heads"]
        kv = sz["head_kv"]
        hv = sz["head_v"]
        df = sz["d_ffn"]
        self.emb = nn.Embedding(CFG["vocab_size"], d, padding_idx=0)
        self.layers = nn.ModuleList([GLABlock(d, h, kv, hv, df, CFG["rms_norm_eps"]) for _ in range(n)])
        self.norm_f = RMSNorm(d, CFG["rms_norm_eps"])
        self.head = nn.Linear(d, CFG["vocab_size"], bias=False)
        nn.init.normal_(self.emb.weight, 0, 0.02)
        self.head.weight = self.emb.weight
        self.use_ckpt = CFG["grad_ckpt"]

    def forward(self, tok):
        x = self.emb(tok)
        for i in range(len(self.layers)):
            if self.use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(self.layers[i], x, use_reentrant=False)
            else:
                x = self.layers[i](x)
        return self.head(self.norm_f(x))


def build_optimizer(model):
    decay, no_decay, seen = [], [], set()
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 1 or any(k in n.lower() for k in ("bias", "norm", "emb")):
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


@torch.no_grad()
def eval_bpb(model, loader, max_batches=None, distributed=True):
    model.eval()
    nll, tok = 0.0, 0
    max_batches = max_batches if max_batches else CFG["eval_batches"]
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = b.to(DEVICE, non_blocking=True)
        x, y = b[:, :-1], b[:, 1:]
        with autocast():
            lo = model(x)
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


def train_one(sz, train_dl, val_dl, ckpt_dir):
    model = GLALM(sz).to(DEVICE)
    if IS_DIST:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK] if DEVICE.type == "cuda" else None,
            output_device=LOCAL_RANK if DEVICE.type == "cuda" else None,
            find_unused_parameters=False,
        )
    core = model.module if IS_DIST else model

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

    it = iter(train_dl)
    g = 0
    tokens_seen = 0
    next_val = CFG["tokens_between_val"]
    best_bpb = float("inf")

    while tokens_seen < target_tokens:
        model.train()
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
                logits = model(x)
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
            vbpb = eval_bpb(model, val_dl, max_batches=CFG["eval_batches"])
            emit(f"K={sz['K']} val={vbpb:.4f}")
            if vbpb < best_bpb - 1e-4:
                best_bpb = vbpb
                if IS_MASTER and ckpt_dir is not None:
                    atomic_save({"model": core.state_dict(), "sz": sz, "implementation": "gla_recurrence_v2"}, str(ckpt_dir / f"K{sz['K']}_best.pt"))

    return model, core


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
    pa.add_argument("--inference-only", action="store_true")
    pa.add_argument("--num-workers", type=int, default=None)
    pa.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = pa.parse_args()
    if args.num_workers is not None:
        if args.num_workers < 0:
            pa.error("--num-workers must be nonnegative")
        CFG["num_workers"] = args.num_workers

    maybe_init_dist()
    set_seed(CFG["seed"] + RANK)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    val_t = build_cache(args.data_dir, args.output_dir, "validation", CFG["val_segments"], CFG["seed"] + 111)
    test_t = build_cache(args.data_dir, args.output_dir, "test", CFG["test_segments"], CFG["seed"] + 222)
    train_dl = None
    if not args.inference_only:
        train_ds = StreamingByteDataset(args.data_dir, "train", CFG["seed"])
        train_dl = DataLoader(
            train_ds, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"],
            pin_memory=True, collate_fn=torch.stack,
            drop_last=True, persistent_workers=CFG["num_workers"] > 0,
        )
    val_dl = DataLoader(CachedSegments(val_t), batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    test_dl = DataLoader(CachedSegments(test_t), batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for sz in SIZES:
        if not args.inference_only:
            model, core = train_one(sz, train_dl, val_dl, ckpt_dir)
        else:
            model = GLALM(sz).to(DEVICE)
            core = model
        if IS_DIST:
            dist.barrier()
        bp = ckpt_dir / f"K{sz['K']}_best.pt"
        if not bp.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {bp}")
        ck = torch.load(str(bp), map_location="cpu", weights_only=True)
        if ck.get("implementation") != "gla_recurrence_v2":
            raise ValueError("Legacy GLA checkpoint has no verified corrected-recurrence provenance; audit its training source or retrain.")
        core.load_state_dict(ck["model"])
        del ck
        if IS_MASTER:
            core.eval()
            v = eval_bpb(core, val_dl, max_batches=CFG["eval_batches"], distributed=False)
            t = eval_bpb(core, test_dl, max_batches=CFG["test_batches"], distributed=False)
            emit(f"K={sz['K']} val={v:.4f} test={t:.4f}")
        del model, core
        cleanup()

    if IS_DIST:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
