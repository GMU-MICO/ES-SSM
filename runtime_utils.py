import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from filelock import FileLock


def parquet_files(data_dir, split):
    if not data_dir:
        raise ValueError("A local dataset directory is required.")
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Local dataset directory does not exist: {root}")
    files = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet shards under {root / 'data'}")
    return files


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=path.parent) as tmp:
        pending = Path(tmp) / path.name
        torch.save(value, pending)
        os.replace(pending, path)


def _metadata(data_dir, split, n, seed, seq_len):
    return {
        "version": 1, "split": split, "seed": int(seed),
        "seq_len": int(seq_len), "segments": int(n),
        "encoding": "utf8-ignore+LF;byte-offset=2;overlap=1",
        "files": [{"name": p.name, "sha256": file_sha256(p)} for p in parquet_files(data_dir, split)],
    }


def _tensor_sha256(t):
    return hashlib.sha256(t.numpy().astype("<i2", copy=False).tobytes()).hexdigest()


def _load_cache(path, metadata):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("metadata") != metadata:
        raise ValueError(f"Cache provenance is missing or mismatched: {path}. Use a new output directory or move the old cache aside.")
    t = value.get("t")
    if (
        not isinstance(t, torch.Tensor) or t.dtype != torch.int16
        or tuple(t.shape) != (metadata["segments"], metadata["seq_len"] + 1)
        or int(t.min()) < 2 or int(t.max()) > 257
        or value.get("tensor_sha256") != _tensor_sha256(t)
    ):
        raise ValueError(f"Cache tensor failed integrity validation: {path}")
    return t


def _segments(data_dir, split, seed, seq_len):
    files = parquet_files(data_dir, split)
    rng = random.Random(seed)
    buf = bytearray()
    while True:
        order = list(files)
        rng.shuffle(order)
        nonempty = False
        for path in order:
            for rb in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
                for text in rb.column(0).to_pylist():
                    if not text:
                        continue
                    nonempty = True
                    buf.extend(text.encode("utf-8", errors="ignore"))
                    buf.extend(b"\n")
                    while len(buf) >= seq_len + 1:
                        yield torch.frombuffer(bytearray(buf[:seq_len + 1]), dtype=torch.uint8).long().add_(2)
                        del buf[:seq_len]
        if not nonempty:
            raise ValueError(f"No nonempty text in the {split} split.")


@torch.no_grad()
def build_eval_cache(data_dir, cache_dir, split, n, seed, seq_len, rank=0, world_size=1, distributed=False):
    if n <= 0 or seq_len <= 0:
        raise ValueError("Cache size and sequence length must be positive.")
    root = Path(cache_dir)
    path = root / f"slimpj_{split}_L{seq_len}_N{n}.pt"
    status, tensor = [None], None
    if not distributed or rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock", timeout=300):
                metadata = _metadata(data_dir, split, n, seed, seq_len)
                if path.exists():
                    tensor = _load_cache(path, metadata)
                else:
                    stream = iter(_segments(data_dir, split, seed, seq_len))
                    tensor = torch.stack([next(stream).to(torch.int16) for _ in range(n)])
                    if _metadata(data_dir, split, n, seed, seq_len) != metadata:
                        raise RuntimeError("Dataset changed during cache generation.")
                    atomic_save({"t": tensor, "metadata": metadata, "tensor_sha256": _tensor_sha256(tensor)}, path)
                status[0] = {"metadata": metadata}
        except Exception as exc:
            if not distributed:
                raise
            status[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        dist.broadcast_object_list(status, src=0)
        if "error" in status[0]:
            raise RuntimeError(status[0]["error"])
        error = None
        if rank != 0:
            try:
                tensor = _load_cache(path, status[0]["metadata"])
            except Exception as exc:
                error = f"Rank {rank}: {type(exc).__name__}: {exc}"
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError("; ".join(x for x in errors if x))
    return tensor


def hankel_kernel(length, eps=1e-6):
    s = torch.arange(2, 2 * length + 1, dtype=torch.float32, device="cpu")
    a = 2 / (s ** 3 - s + eps)
    return torch.cat([a[length - 1:], torch.zeros(1), a[:length - 1]])


def hmv(g_f, x):
    length, p = x.shape
    padded = torch.zeros(p, 2 * length, dtype=x.dtype, device=x.device)
    padded[:, :length] = x.flip(0).T
    return torch.fft.irfft(torch.fft.rfft(padded, dim=-1) * g_f[None], n=2 * length, dim=-1)[:, :length].T


def _basis_digest(sigma, phi):
    digest = hashlib.sha256()
    for tensor in (sigma, phi):
        digest.update(tensor.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


@torch.no_grad()
def hankel_basis(cache_dir, length, channels, iters=60, oversample=16, tol=1e-6):
    if not 1 <= channels <= length or iters <= 0 or oversample < 0 or tol <= 0:
        raise ValueError("Invalid Hankel basis configuration.")
    metadata = dict(version=2, length=length, channels=channels, iters=iters, oversample=oversample, tol=tol, seed=0)
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"hankel_v2_L{length}_K{channels}_{signature}.pt"
    with FileLock(str(path) + ".lock", timeout=300):
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            sigma, phi = saved["sigma"], saved["phi"]
            if (
                saved.get("metadata") != metadata or sigma.shape != (channels,)
                or phi.shape != (length, channels)
                or not torch.isfinite(sigma).all() or not torch.isfinite(phi).all()
                or saved.get("sha256") != _basis_digest(sigma, phi)
            ):
                raise ValueError(f"Invalid Hankel cache: {path}")
            return sigma, phi
        generator = torch.Generator(device="cpu").manual_seed(0)
        q = torch.randn(length, channels + oversample, generator=generator, device="cpu", dtype=torch.float32)
        q, _ = torch.linalg.qr(q, mode="reduced")
        gf = torch.fft.rfft(hankel_kernel(length))
        for step in range(1, iters + 1):
            q, _ = torch.linalg.qr(hmv(gf, q), mode="reduced")
            if step % max(1, iters // 4) == 0 or step == iters:
                values, vectors = torch.linalg.eigh(q.T @ hmv(gf, q))
                order = values.argsort(descending=True)
                values, vectors = values[order], vectors[:, order]
                q = q @ vectors
                residual = (hmv(gf, q[:, :channels]) - q[:, :channels] * values[:channels][None]).norm(dim=0)
                if float((residual / (values[:channels].abs() + 1e-12)).max()) < tol:
                    break
        values, vectors = torch.linalg.eigh(q.T @ hmv(gf, q))
        order = values.argsort(descending=True)
        sigma, phi = values[order][:channels], (q @ vectors[:, order])[:, :channels]
        atomic_save({"sigma": sigma, "phi": phi, "metadata": metadata, "sha256": _basis_digest(sigma, phi)}, path)
        return sigma, phi
