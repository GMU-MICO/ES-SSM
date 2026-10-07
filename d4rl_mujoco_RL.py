from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from runtime_utils import atomic_save, hankel_basis


REF_SCORES = {
    "hopper":      {"rand": -20.272,  "expert": 3234.3},
    "halfcheetah": {"rand": -280.178, "expert": 12135.0},
    "walker2d":    {"rand": 1.629,    "expert": 4592.3},
}

ENV_CFG = {
    "hopper":      dict(targets=(3600, 1800),   scale=1000.0, gym_id="Hopper-v4"),
    "halfcheetah": dict(targets=(12000, 6000),  scale=1000.0, gym_id="HalfCheetah-v4"),
    "walker2d":    dict(targets=(5000, 2500),   scale=1000.0, gym_id="Walker2d-v4"),
}

DATASETS = [
    ("halfcheetah", "medium",        "halfcheetah_medium-v2.hdf5"),
    ("halfcheetah", "medium-replay", "halfcheetah_medium_replay-v2.hdf5"),
    ("halfcheetah", "medium-expert", "halfcheetah_medium_expert-v2.hdf5"),
    ("hopper",      "medium",        "hopper_medium-v2.hdf5"),
    ("hopper",      "medium-replay", "hopper_medium_replay-v2.hdf5"),
    ("hopper",      "medium-expert", "hopper_medium_expert-v2.hdf5"),
    ("walker2d",    "medium",        "walker2d_medium-v2.hdf5"),
    ("walker2d",    "medium-replay", "walker2d_medium_replay-v2.hdf5"),
    ("walker2d",    "medium-expert", "walker2d_medium_expert-v2.hdf5"),
]

D4RL_HDF5_URL_BASE = (
    "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"
)

DEFAULT_HYPER = dict(
    embed_dim=128,
    n_layer=3,
    n_head=1,
    K_ctx=20,
    batch_size=64,
    lr=1e-4,
    weight_decay=1e-4,
    dropout=0.1,
    activation="relu",
    warmup_steps=10_000,
    num_steps=150_000,
    num_eval_episodes_train=10,
    grad_clip=0.25,
    max_ep_len=1024,
    seed=42,
    K_max=32,
    budget_warmup_frac=0.10,
    budget_full_every=8,
    budget_K_min=2,
    budget_ks=(2, 3, 4, 5, 6, 8, 12, 16, 24, 32),
)

ALL_TAGS = (
    "dt",
    "esssm_K32",
    "tf_matchK2",
    "tf_matchK4",
    "tf_matchK8",
    "tf_matchK16",
)
DEFAULT_SEEDS = (42, 43, 44)
COMPACT_KS = (2, 4, 8, 16, 32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def normalized_score(env_name: str, raw_return: float) -> float:
    r = REF_SCORES[env_name]
    return (raw_return - r["rand"]) / (r["expert"] - r["rand"]) * 100.0


def dataset_file(hdf5_dir: Path, env: str, dataset: str) -> Path:
    suffix = {"medium": "medium",
              "medium-replay": "medium_replay",
              "medium-expert": "medium_expert"}[dataset]
    return hdf5_dir / f"{env}_{suffix}-v2.hdf5"


def make_env(env_name: str):
    import gymnasium as gym
    return gym.make(ENV_CFG[env_name]["gym_id"])


def cmd_download_data(args):
    out = Path(args.hdf5_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [fname for _, _, fname in DATASETS]
    targets = [(out / f, f"{D4RL_HDF5_URL_BASE}/{f}") for f in files]
    todo = [(p, u) for p, u in targets if not p.exists()]
    if not todo:
        print(f"[data] all 9 files already present in {out}")
        return
    print(f"[data] downloading {len(todo)} files to {out}")
    procs = []
    for p, u in todo:
        cmd = ["curl", "-fL", "-o", str(p) + ".part", u]
        procs.append((p, subprocess.Popen(cmd)))
    for p, proc in procs:
        rc = proc.wait()
        if rc == 0:
            os.replace(str(p) + ".part", str(p))
            sz = p.stat().st_size / 1e6
            print(f"  [ok] {p.name}  {sz:.1f} MB")
        else:
            print(f"  [FAIL] {p.name}  curl rc={rc}")
            sys.exit(1)
    print(f"[data] done. total: {sum(p.stat().st_size for p, _ in targets)/1e9:.2f} GB")


def load_trajectories(hdf5_path: Path) -> List[Dict[str, np.ndarray]]:
    import h5py
    with h5py.File(str(hdf5_path), "r") as h:
        obs = h["observations"][:]
        act = h["actions"][:]
        rew = h["rewards"][:].astype(np.float32)
        term = h["terminals"][:].astype(bool)
        tout = h["timeouts"][:].astype(bool) if "timeouts" in h else np.zeros_like(term)
    done = np.logical_or(term, tout)
    ends = np.where(done)[0].tolist()
    if not ends or ends[-1] != len(rew) - 1:
        ends.append(len(rew) - 1)
    starts = [0] + [e + 1 for e in ends[:-1]]
    trajs = []
    for s, e in zip(starts, ends):
        trajs.append({
            "observations": obs[s:e + 1],
            "actions":      act[s:e + 1],
            "rewards":      rew[s:e + 1],
            "terminals":    term[s:e + 1],
        })
    return trajs


def compute_state_norm(trajs: List[Dict[str, np.ndarray]]):
    stack = np.concatenate([t["observations"] for t in trajs], axis=0).astype(np.float32)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def returns_to_go(rew: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    rtg = np.zeros_like(rew, dtype=np.float32)
    acc = 0.0
    for t in range(len(rew) - 1, -1, -1):
        acc = rew[t] + gamma * acc
        rtg[t] = acc
    return rtg


class TrajectoryBatcher:
    def __init__(self, trajs, K, state_dim, act_dim, scale,
                 state_mean, state_std, max_ep_len, pct_traj=1.0, seed=0):
        self.trajs = trajs
        self.K = K
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_ep_len = max_ep_len
        self.rng = np.random.RandomState(seed)
        self._rtg = [returns_to_go(t["rewards"]) for t in trajs]
        traj_lens = np.array([len(t["rewards"]) for t in trajs], dtype=np.int64)
        returns = np.array([t["rewards"].sum() for t in trajs], dtype=np.float32)
        num_timesteps = int(max(int(pct_traj * traj_lens.sum()), 1))
        sorted_inds = np.argsort(returns)
        timesteps = int(traj_lens[sorted_inds[-1]])
        num_traj = 1
        i = len(traj_lens) - 2
        while i >= 0 and timesteps + traj_lens[sorted_inds[i]] <= num_timesteps:
            timesteps += int(traj_lens[sorted_inds[i]])
            num_traj += 1
            i -= 1
        self.sorted_inds = sorted_inds[-num_traj:]
        self.p_sample = (traj_lens[self.sorted_inds] /
                         traj_lens[self.sorted_inds].sum()).astype(np.float64)

    def sample(self, batch_size: int):
        K, sd, ad = self.K, self.state_dim, self.act_dim
        batch_inds = self.rng.choice(len(self.sorted_inds), size=batch_size,
                                     replace=True, p=self.p_sample)
        s_arr = np.zeros((batch_size, K, sd), dtype=np.float32)
        a_arr = np.full((batch_size, K, ad), -10.0, dtype=np.float32)
        rtg_arr = np.zeros((batch_size, K, 1), dtype=np.float32)
        ts_arr = np.zeros((batch_size, K), dtype=np.int64)
        mask_arr = np.zeros((batch_size, K), dtype=np.float32)
        for i, bi in enumerate(batch_inds):
            tid = int(self.sorted_inds[bi])
            traj = self.trajs[tid]
            T = traj["rewards"].shape[0]
            si = random.randint(0, T - 1)
            ei = min(si + K, T)
            tlen = ei - si
            pad = K - tlen
            s_slice = (traj["observations"][si:ei] - self.state_mean) / self.state_std
            s_arr[i, pad:] = s_slice.astype(np.float32)
            a_arr[i, pad:] = traj["actions"][si:ei].astype(np.float32)
            rtg_arr[i, pad:, 0] = self._rtg[tid][si:ei] / self.scale
            ts = np.arange(si, si + tlen)
            ts = np.minimum(ts, self.max_ep_len - 1)
            ts_arr[i, pad:] = ts.astype(np.int64)
            mask_arr[i, pad:] = 1.0
        return {
            "states":    torch.as_tensor(s_arr, dtype=torch.float32),
            "actions":   torch.as_tensor(a_arr, dtype=torch.float32),
            "rtg":       torch.as_tensor(rtg_arr, dtype=torch.float32),
            "timesteps": torch.as_tensor(ts_arr, dtype=torch.long),
            "mask":      torch.as_tensor(mask_arr, dtype=torch.float32),
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_head: int, p_drop: float):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.d_head = d // n_head
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(p_drop)
        self.resid_drop = nn.Dropout(p_drop)

    def forward(self, x, attn_mask):
        B, T, D = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scale = 1.0 / math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            key_mask = attn_mask[:, None, None, :].to(q.dtype)
            scores = scores.masked_fill(key_mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_drop(self.c_proj(y))


class GPTBlock(nn.Module):
    def __init__(self, d, n_head, p_drop, activation, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, p_drop)
        self.ln2 = nn.LayerNorm(d)
        act = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]
        self.mlp = nn.Sequential(
            nn.Linear(d, ffn_mult * d), act(),
            nn.Linear(ffn_mult * d, d), nn.Dropout(p_drop),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, d, n_layer, n_head, p_drop, activation="relu"):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GPTBlock(d, n_head, p_drop, activation) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x, attn_mask=None):
        for b in self.blocks:
            x = b(x, attn_mask)
        return self.ln_f(x)


def _hankel_g(L, eps=1e-6):
    k = torch.arange(0, 2 * L - 1, dtype=torch.float32)
    s = k + 2
    a = 2 / (s ** 3 - s + eps)
    return torch.cat([a[L - 1:], torch.zeros(1), a[: L - 1]])


@torch.no_grad()
def _hmv(g_f, X):
    L, p = X.shape
    n = 2 * L
    Xp = torch.zeros((p, n))
    Xp[:, :L] = torch.flip(X, [0]).T
    return torch.fft.irfft(torch.fft.rfft(Xp, dim=-1) * g_f.unsqueeze(0),
                           n=n, dim=-1)[:, :L].T


@torch.no_grad()
def hankel_topk(L: int, K: int, cache_dir: str,
                iters: int = 60, oversample: int = 16, tol: float = 1e-6):
    return hankel_basis(cache_dir, L, K, iters, oversample, tol)


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = torch.rand(shape, device=x.device, dtype=x.dtype) < (1 - self.p)
        return x * keep / (1 - self.p)


class ESSMLayer(nn.Module):
    def __init__(self, d, L, K, gate_bias_init=-2.0,
                 drop_path=0.0, cache_dir="./cache_filters",
                 m_init_scale=1e-3):
        super().__init__()
        self.K = K
        self.L = L
        self.d = d
        sigma, phi = hankel_topk(L, K, cache_dir)
        self.register_buffer("sigma", sigma)
        with torch.no_grad():
            phi_f = torch.fft.rfft(phi.T.float(), n=2 * L, dim=-1)
        self.register_buffer("phi_f_ri", torch.view_as_real(phi_f).contiguous())
        self.M = nn.Parameter(torch.randn(K, d, d) * m_init_scale)
        self.sel = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, K),
        )
        nn.init.constant_(self.sel[-1].bias, float(gate_bias_init))
        self.dp = DropPath(drop_path)

    def forward(self, u, K_rt: Optional[int] = None, K_min: int = 2):
        B, L, D = u.shape
        n = 2 * L
        K = self.K if K_rt is None else max(K_min, min(self.K, int(K_rt)))
        w = (torch.sigmoid(self.sel(u)[:, :, :K].float()) / math.sqrt(K)).to(u.dtype)
        device_type = "cuda" if u.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            xf = torch.fft.rfft(u.float().permute(0, 2, 1), n=n, dim=-1)
            pf = torch.view_as_complex(self.phi_f_ri[:K].contiguous())
            if L <= 256:
                Uo = torch.fft.irfft(
                    xf.unsqueeze(2) * pf.unsqueeze(0).unsqueeze(0),
                    n=n, dim=-1,
                )[..., :L]
                Uo = Uo.permute(0, 3, 2, 1).contiguous().to(u.dtype)
                y = torch.einsum("blkd,kdo->blo",
                                 Uo * w.unsqueeze(-1), self.M[:K].to(u.dtype))
            else:
                M_full = self.M[:K].to(u.dtype)
                y = torch.zeros(B, L, D, device=u.device, dtype=u.dtype)
                for k in range(K):
                    u_k = torch.fft.irfft(xf * pf[k], n=n, dim=-1)[..., :L]
                    u_k = u_k.permute(0, 2, 1).contiguous().to(u.dtype)
                    y = y + (u_k * w[..., k:k + 1]) @ M_full[k]
        return self.dp(y)


class ESSSMBackbone(nn.Module):
    def __init__(self, d, L, n_layer, K_max, p_drop, stoch_depth,
                 gate_bias_init=-2.0, cache_dir="./cache_filters"):
        super().__init__()
        self.d = d
        self.L = L
        self.K_max = K_max
        self.n_layer = n_layer
        sd = torch.linspace(0, stoch_depth, n_layer).tolist()
        self.layers = nn.ModuleList([
            ESSMLayer(d, L, K_max, gate_bias_init, sd[i], cache_dir)
            for i in range(n_layer)
        ])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(p_drop),
                nn.Linear(d * 4, d),
            ) for _ in range(n_layer)
        ])
        self.ffn_dp = nn.ModuleList([DropPath(sd[i]) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x, K_rt: Optional[int] = None):
        for i in range(self.n_layer):
            y = self.layers[i](self.n1[i](x), K_rt=K_rt)
            x = x + y
            x = x + self.ffn_dp[i](self.ffn[i](self.n2[i](x)))
        return self.ln_f(x)


@dataclass
class RLModelConfig:
    state_dim: int
    act_dim: int
    max_ep_len: int = 1024
    K_ctx: int = 20
    d_model: int = 128
    dropout: float = 0.1
    action_tanh: bool = True


class RLSequenceModel(nn.Module):
    def __init__(self, cfg: RLModelConfig, backbone: nn.Module, backbone_kind: str):
        super().__init__()
        self.cfg = cfg
        self.backbone_kind = backbone_kind
        d = cfg.d_model
        self.embed_state = nn.Linear(cfg.state_dim, d)
        self.embed_action = nn.Linear(cfg.act_dim, d)
        self.embed_return = nn.Linear(1, d)
        self.embed_timestep = nn.Embedding(cfg.max_ep_len, d)
        self.embed_ln = nn.LayerNorm(d)
        head = [nn.Linear(d, cfg.act_dim)]
        if cfg.action_tanh:
            head.append(nn.Tanh())
        self.predict_action = nn.Sequential(*head)
        self.backbone = backbone

    def _stack_tokens(self, states, actions, rtg, timesteps):
        B, T, _ = states.shape
        t_emb = self.embed_timestep(timesteps)
        s_emb = self.embed_state(states) + t_emb
        a_emb = self.embed_action(actions) + t_emb
        r_emb = self.embed_return(rtg) + t_emb
        stacked = torch.stack((r_emb, s_emb, a_emb), dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, -1)
        return self.embed_ln(stacked)

    def forward(self, states, actions, rtg, timesteps, mask, K_rt: Optional[int] = None):
        B, T = states.shape[0], states.shape[1]
        tokens = self._stack_tokens(states, actions, rtg, timesteps)
        tok_mask = mask.unsqueeze(-1).expand(B, T, 3).reshape(B, 3 * T).to(tokens.dtype)
        if self.backbone_kind == "transformer":
            h = self.backbone(tokens, attn_mask=tok_mask)
        else:
            h = self.backbone(tokens, K_rt=K_rt)
        h = h.reshape(B, T, 3, -1).permute(0, 2, 1, 3)
        return self.predict_action(h[:, 1])

    @torch.no_grad()
    def act(self, states, actions, rtg, timesteps, K_rt: Optional[int] = None):
        K = self.cfg.K_ctx
        states = states[:, -K:]
        actions = actions[:, -K:]
        rtg = rtg[:, -K:]
        timesteps = timesteps[:, -K:]
        T = states.shape[1]
        pad = K - T
        device = states.device
        if pad > 0:
            s_pad = torch.zeros((1, pad, self.cfg.state_dim), device=device, dtype=states.dtype)
            a_pad = torch.zeros((1, pad, self.cfg.act_dim), device=device, dtype=actions.dtype)
            r_pad = torch.zeros((1, pad, 1), device=device, dtype=rtg.dtype)
            t_pad = torch.zeros((1, pad), device=device, dtype=timesteps.dtype)
            states = torch.cat([s_pad, states], dim=1)
            actions = torch.cat([a_pad, actions], dim=1)
            rtg = torch.cat([r_pad, rtg], dim=1)
            timesteps = torch.cat([t_pad, timesteps], dim=1)
            mask = torch.cat([torch.zeros(1, pad, device=device),
                              torch.ones(1, T, device=device)], dim=1)
        else:
            mask = torch.ones(1, K, device=device)
        a = self.forward(states, actions, rtg, timesteps, mask, K_rt=K_rt)
        return a[0, -1]


def build_transformer_model(cfg: RLModelConfig, n_layer: int, n_head: int,
                            activation: str = "relu") -> RLSequenceModel:
    backbone = TransformerBackbone(cfg.d_model, n_layer, n_head, cfg.dropout, activation)
    return RLSequenceModel(cfg, backbone, backbone_kind="transformer")


def build_esssm_model(cfg: RLModelConfig, n_layer: int, K_max: int,
                     stoch_depth: float = 0.0, gate_bias_init: float = -2.0,
                     cache_dir: str = "./cache_filters") -> RLSequenceModel:
    L = 3 * cfg.K_ctx
    backbone = ESSSMBackbone(
        d=cfg.d_model, L=L, n_layer=n_layer, K_max=K_max,
        p_drop=cfg.dropout, stoch_depth=stoch_depth,
        gate_bias_init=gate_bias_init, cache_dir=cache_dir,
    )
    return RLSequenceModel(cfg, backbone, backbone_kind="esssm")


def _build_for_count(kind, d, n_layer, cfg, K_max, n_head, cache_dir):
    new_cfg = RLModelConfig(**{**cfg.__dict__, "d_model": d})
    if kind == "transformer":
        m = build_transformer_model(new_cfg, n_layer=n_layer, n_head=n_head)
    else:
        m = build_esssm_model(new_cfg, n_layer=n_layer, K_max=K_max,
                             cache_dir=cache_dir)
    n = count_params(m)
    del m
    return n


def match_dim_for_target(target_params, kind, n_layer, cfg, cache_dir,
                         K_max=32, n_head=1, lo=16, hi=1024, div_by=4):
    step = max(div_by, n_head)

    def snap(d):
        d = max(step, d - (d % step))
        if kind == "esssm" and d % 2:
            d += 1
        return d

    l, r = snap(lo), snap(hi)
    while _build_for_count(kind, r, n_layer, cfg, K_max, n_head, cache_dir) < target_params and r < 4096:
        r = snap(r * 2)
    best_d = l
    best_diff = abs(_build_for_count(kind, l, n_layer, cfg, K_max, n_head, cache_dir) - target_params)
    while l <= r:
        mid = snap((l + r) // 2)
        n = _build_for_count(kind, mid, n_layer, cfg, K_max, n_head, cache_dir)
        diff = abs(n - target_params)
        if diff < best_diff:
            best_diff = diff
            best_d = mid
        if n < target_params:
            l = snap(mid + step)
        else:
            r = snap(mid - step)
        if l == r:
            break
    final_n = _build_for_count(kind, best_d, n_layer, cfg, K_max, n_head, cache_dir)
    return int(best_d), int(final_n)


def build_compact_esssm_from_prefix(source: RLSequenceModel, K: int,
                                    cache_dir: str) -> RLSequenceModel:
    assert source.backbone_kind == "esssm"
    src_bb: ESSSMBackbone = source.backbone
    assert 1 <= K <= src_bb.K_max
    new_cfg = RLModelConfig(**{**source.cfg.__dict__})
    compact = build_esssm_model(new_cfg, n_layer=src_bb.n_layer, K_max=K,
                               stoch_depth=0.0, cache_dir=cache_dir)
    with torch.no_grad():
        compact.embed_state.load_state_dict(source.embed_state.state_dict())
        compact.embed_action.load_state_dict(source.embed_action.state_dict())
        compact.embed_return.load_state_dict(source.embed_return.state_dict())
        compact.embed_timestep.load_state_dict(source.embed_timestep.state_dict())
        compact.embed_ln.load_state_dict(source.embed_ln.state_dict())
        compact.predict_action.load_state_dict(source.predict_action.state_dict())
        compact.backbone.ln_f.load_state_dict(src_bb.ln_f.state_dict())
        for i in range(src_bb.n_layer):
            compact.backbone.n1[i].load_state_dict(src_bb.n1[i].state_dict())
            compact.backbone.n2[i].load_state_dict(src_bb.n2[i].state_dict())
            compact.backbone.ffn[i].load_state_dict(src_bb.ffn[i].state_dict())
            s_layer = src_bb.layers[i]
            d_layer = compact.backbone.layers[i]
            d_layer.sigma.copy_(s_layer.sigma[:K].detach().clone())
            d_layer.phi_f_ri.copy_(s_layer.phi_f_ri[:K].detach().clone())
            d_layer.M.copy_(s_layer.M[:K].detach().clone())
            d_layer.sel[0].load_state_dict(s_layer.sel[0].state_dict())
            d_layer.sel[2].weight.copy_(s_layer.sel[2].weight[:K].detach().clone())
            d_layer.sel[2].bias.copy_(s_layer.sel[2].bias[:K].detach().clone())
    return compact


class Budget:
    def __init__(self, K_max=32, budget_ks=None, K_min=2,
                 full_every=8, warmup_steps=0, seed=0):
        self._rng = random.Random(seed)
        self.K_max = int(K_max)
        self.full_every = int(full_every)
        self.warmup_steps = int(warmup_steps)
        ks = sorted(budget_ks or [2, 3, 4, 5, 6, 8, 12, 16, 24, 32])
        self.eligible = [k for k in ks if k >= K_min]
        self.lo = max(K_min, min(ks))
        self.hi = max(ks)
        self.log_ratio = math.log(self.hi / self.lo)

    def sample(self, step: int) -> int:
        if step < self.warmup_steps:
            return self.K_max
        if self.full_every > 0 and step % self.full_every == 0:
            return self.K_max
        u = self._rng.random()
        target = self.lo * math.exp(u * self.log_ratio)
        return min(self.eligible, key=lambda k: abs(k - target))


def _lr_schedule(num_steps: int, warmup_steps: int):
    def lam(s):
        return min((s + 1) / max(1, warmup_steps), 1.0)
    return lam


def train_loop(model, batcher, *, num_steps, batch_size, lr, weight_decay,
              warmup_steps, device, eval_every, eval_fn=None, grad_clip=0.25,
              progress_path=None, budget: Optional[Budget] = None):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_schedule(num_steps, warmup_steps))
    is_elastic = (budget is not None and getattr(model, "backbone_kind", "") == "esssm")
    hist = []
    t0 = time.time()
    nll_acc = 0.0
    acc_n = 0
    best_norm = -float("inf")
    for step in range(1, num_steps + 1):
        batch = batcher.sample(batch_size)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        K_rt = budget.sample(step) if is_elastic else None
        a_pred = model(batch["states"], batch["actions"], batch["rtg"],
                       batch["timesteps"], batch["mask"], K_rt=K_rt)
        a_tgt = batch["actions"]
        mask = batch["mask"].unsqueeze(-1)
        err = ((a_pred - a_tgt) ** 2) * mask
        denom = mask.sum().clamp(min=1.0) * a_tgt.shape[-1]
        loss = err.sum() / denom
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        nll_acc += float(loss.item())
        acc_n += 1
        if step % 200 == 0:
            cur = nll_acc / max(1, acc_n)
            nll_acc, acc_n = 0.0, 0
            elapsed = time.time() - t0
            line = {"kind": "train", "step": step, "loss": cur,
                    "lr": opt.param_groups[0]["lr"], "elapsed_s": round(elapsed, 1)}
            hist.append(line)
            print(f"  step={step:>6d}  loss={cur:.5f}  lr={opt.param_groups[0]['lr']:.2e}"
                  f"  elapsed={elapsed:.1f}s", flush=True)
            if progress_path is not None:
                with open(progress_path, "a") as f:
                    f.write(json.dumps(line) + "\n")
        if eval_fn is not None and step % eval_every == 0:
            model.eval()
            metrics = eval_fn(model)
            model.train()
            row = {"kind": "eval", "step": step, **metrics}
            hist.append(row)
            print(f"  [eval@{step}] " + "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()), flush=True)
            if progress_path is not None:
                with open(progress_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            norm_candidates = [v for k, v in metrics.items() if "norm_mean" in k]
            if norm_candidates:
                best_norm = max(best_norm, max(norm_candidates))
    total_s = time.time() - t0
    return {"history": hist, "total_train_s": total_s, "best_norm": best_norm}


@torch.no_grad()
def evaluate_episode_rtg(env, state_dim, act_dim, model, *, max_ep_len, scale,
                        target_return, state_mean, state_std, device,
                        K_rt: Optional[int] = None, seed: Optional[int] = None):
    model.eval()
    s_mean = torch.as_tensor(state_mean, device=device, dtype=torch.float32)
    s_std = torch.as_tensor(state_std, device=device, dtype=torch.float32)
    reset_kwargs = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kwargs)
    state = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                            device=device).reshape(1, state_dim)
    states = state
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rtg = torch.tensor([[target_return]], device=device, dtype=torch.float32)
    timesteps = torch.tensor([0], device=device, dtype=torch.long)
    a_low = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
    a_high = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)
    ep_return = 0.0
    ep_len = 0
    context_length = model.cfg.K_ctx
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)[-context_length:]
        s_norm = (states - s_mean) / s_std
        action = model.act(
            states=s_norm.reshape(1, -1, state_dim),
            actions=actions.reshape(1, -1, act_dim),
            rtg=rtg.reshape(1, -1, 1),
            timesteps=timesteps.reshape(1, -1),
            K_rt=K_rt,
        )
        action = torch.clamp(action, a_low, a_high)
        actions[-1] = action.to(dtype=actions.dtype)
        action_np = np.asarray(action.detach().cpu().tolist(), dtype=np.float32)
        obs, rew, terminated, truncated, _ = env.step(action_np)
        ep_return += float(rew)
        ep_len += 1
        new_state = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=device).reshape(1, state_dim)
        states = torch.cat([states, new_state], dim=0)[-context_length:]
        pred_rtg = rtg[0, -1] - (float(rew) / scale)
        rtg = torch.cat([rtg, pred_rtg.reshape(1, 1)], dim=1)[:, -context_length:]
        timesteps = torch.cat([timesteps,
                               torch.tensor([min(t + 1, max_ep_len - 1)],
                                            device=device, dtype=torch.long)])[-context_length:]
        if terminated or truncated:
            break
    return ep_return, ep_len


def rollout_many(env, state_dim, act_dim, model, *, max_ep_len, scale,
                target_return_scaled, state_mean, state_std, device,
                num_episodes=10, K_rt: Optional[int] = None, base_seed=10000):
    rets, lens = [], []
    for i in range(num_episodes):
        r, l = evaluate_episode_rtg(
            env, state_dim, act_dim, model,
            max_ep_len=max_ep_len, scale=scale,
            target_return=target_return_scaled,
            state_mean=state_mean, state_std=state_std, device=device,
            K_rt=K_rt, seed=base_seed + i,
        )
        rets.append(r)
        lens.append(l)
    return np.asarray(rets, dtype=np.float64), np.asarray(lens, dtype=np.int64)


def summarize_rollouts(env_name: str, returns: np.ndarray):
    norm = np.asarray([normalized_score(env_name, r) for r in returns],
                      dtype=np.float64)
    return {
        "raw_mean": float(returns.mean()),
        "raw_std": float(returns.std()),
        "norm_mean": float(norm.mean()),
        "norm_std": float(norm.std()),
        "num": int(len(returns)),
    }


@torch.no_grad()
def bench_forward(model, dummy_batch, *, K_rt: Optional[int] = None,
                 warmup: int = 5, repeats: int = 30):
    model.eval()
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in dummy_batch.items()}
    for _ in range(warmup):
        _ = model(batch["states"], batch["actions"], batch["rtg"],
                  batch["timesteps"], batch["mask"], K_rt=K_rt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    lats_ms = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(batch["states"], batch["actions"], batch["rtg"],
                  batch["timesteps"], batch["mask"], K_rt=K_rt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lats_ms.append((time.perf_counter() - t0) * 1000.0)
    lats_ms.sort()
    lat_ms = lats_ms[len(lats_ms) // 2]
    B, T = batch["states"].shape[:2]
    tokens_per_step = B * 3 * T
    tput_ktok_s = (tokens_per_step / (lat_ms / 1000.0)) / 1000.0
    cuda_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    return {"lat_ms": float(lat_ms), "tput_ktok_s": float(tput_ktok_s),
            "cuda_mb": cuda_mb, "batch": int(B), "ctx": int(T)}


def _build_model_for_train(args, state_dim, act_dim, max_ep_len, cache_dir):
    cfg = RLModelConfig(state_dim=state_dim, act_dim=act_dim,
                       max_ep_len=max_ep_len, K_ctx=args.K_ctx,
                       d_model=args.embed_dim, dropout=args.dropout,
                       action_tanh=True)
    if args.model == "dt" or (args.model == "transformer" and args.match_params_to is None):
        model = build_transformer_model(cfg, n_layer=args.n_layer,
                                       n_head=args.n_head, activation=args.activation)
        info = {"kind": "transformer", "d": args.embed_dim, "n_layer": args.n_layer}
    elif args.model == "transformer" and args.match_params_to == "esssm_K":
        ref_cfg = RLModelConfig(state_dim=state_dim, act_dim=act_dim,
                               max_ep_len=max_ep_len, K_ctx=args.K_ctx,
                               d_model=args.ref_d_essm, dropout=args.dropout)
        ref_m = build_esssm_model(ref_cfg, n_layer=args.ref_n_layer_essm,
                                 K_max=args.match_K, cache_dir=cache_dir)
        target = count_params(ref_m)
        del ref_m
        best_d, final_n = match_dim_for_target(
            target, kind="transformer", n_layer=args.n_layer, cfg=cfg,
            cache_dir=cache_dir, n_head=args.n_head, K_max=32)
        cfg.d_model = best_d
        model = build_transformer_model(cfg, n_layer=args.n_layer,
                                       n_head=args.n_head, activation=args.activation)
        info = {"kind": "transformer", "d": best_d, "n_layer": args.n_layer,
                "match_K": args.match_K, "target_params": int(target),
                "achieved_params": int(final_n)}
    elif args.model == "esssm":
        if args.match_params_to == "dt":
            dt_cfg = RLModelConfig(state_dim=state_dim, act_dim=act_dim,
                                   max_ep_len=max_ep_len, K_ctx=args.K_ctx,
                                   d_model=DEFAULT_HYPER["embed_dim"], dropout=args.dropout)
            dt_m = build_transformer_model(dt_cfg, n_layer=DEFAULT_HYPER["n_layer"],
                                           n_head=DEFAULT_HYPER["n_head"],
                                           activation=DEFAULT_HYPER["activation"])
            target = count_params(dt_m)
            del dt_m
            best_d, final_n = match_dim_for_target(
                target, kind="esssm", n_layer=args.n_layer, cfg=cfg,
                cache_dir=cache_dir, K_max=args.K_max)
            cfg.d_model = best_d
            model = build_esssm_model(cfg, n_layer=args.n_layer, K_max=args.K_max,
                                     stoch_depth=args.stoch_depth, cache_dir=cache_dir)
            info = {"kind": "esssm", "d": best_d, "n_layer": args.n_layer,
                    "K_max": args.K_max, "target_params": int(target),
                    "achieved_params": int(final_n),
                    "ref": {"d": DEFAULT_HYPER["embed_dim"],
                            "n_layer": DEFAULT_HYPER["n_layer"]}}
        else:
            model = build_esssm_model(cfg, n_layer=args.n_layer, K_max=args.K_max,
                                     stoch_depth=args.stoch_depth, cache_dir=cache_dir)
            info = {"kind": "esssm", "d": args.embed_dim, "n_layer": args.n_layer,
                    "K_max": args.K_max}
    else:
        raise ValueError(f"unsupported (model, match_params_to)="
                         f"({args.model}, {args.match_params_to})")
    return model, cfg, info


def _make_eval_fn(env, env_name, state_dim, act_dim, state_mean, state_std,
                 max_ep_len, scale, targets, device, num_episodes):
    def fn(model):
        out = {}
        for tar in targets:
            rets, lens = rollout_many(
                env, state_dim, act_dim, model,
                max_ep_len=max_ep_len, scale=scale,
                target_return_scaled=tar / scale,
                state_mean=state_mean, state_std=state_std,
                device=device, num_episodes=num_episodes)
            summary = summarize_rollouts(env_name, rets)
            out[f"t{int(tar)}_raw_mean"] = summary["raw_mean"]
            out[f"t{int(tar)}_norm_mean"] = summary["norm_mean"]
            out[f"t{int(tar)}_len_mean"] = float(lens.mean())
        return out
    return fn


def cmd_train(args):
    device = pick_device()
    set_seed(args.seed)
    env_name = args.env
    cfg_env = ENV_CFG[env_name]
    print(f"[cfg] env={env_name} dataset={args.dataset} device={device} model={args.model}",
          flush=True)
    trajs = load_trajectories(dataset_file(Path(args.hdf5_dir), env_name, args.dataset))
    state_mean, state_std = compute_state_norm(trajs)
    state_dim = trajs[0]["observations"].shape[1]
    act_dim = trajs[0]["actions"].shape[1]
    print(f"[data] trajectories={len(trajs)} obs={state_dim} act={act_dim}", flush=True)
    model, cfg, info = _build_model_for_train(args, state_dim, act_dim,
                                              args.max_ep_len, args.cache_dir)
    n_params = count_params(model)
    info["n_params"] = n_params
    print(f"[model] {info}  total_params={n_params:,} ({n_params/1e6:.3f}M)", flush=True)
    batcher = TrajectoryBatcher(
        trajs, K=cfg.K_ctx, state_dim=state_dim, act_dim=act_dim,
        scale=cfg_env["scale"], state_mean=state_mean, state_std=state_std,
        max_ep_len=args.max_ep_len, seed=args.seed)
    env = make_env(env_name)
    eval_fn = _make_eval_fn(env, env_name, state_dim, act_dim,
                           state_mean, state_std, args.max_ep_len,
                           cfg_env["scale"], cfg_env["targets"], device,
                           args.num_eval_episodes_train)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    budget = None
    if args.model == "esssm" and args.budget_enabled:
        budget = Budget(K_max=args.K_max,
                       budget_ks=[int(x) for x in args.budget_ks.split(",")],
                       K_min=args.budget_K_min,
                       full_every=args.budget_full_every,
                       warmup_steps=int(args.budget_warmup_frac * args.num_steps),
                       seed=args.seed + 7777)
        print(f"[budget] warmup_steps={budget.warmup_steps}  "
              f"full_every={budget.full_every}  ks={budget.eligible}", flush=True)
    tres = train_loop(model, batcher,
                     num_steps=args.num_steps, batch_size=args.batch_size,
                     lr=args.lr, weight_decay=args.weight_decay,
                     warmup_steps=args.warmup_steps,
                     device=device, eval_every=args.eval_every,
                     eval_fn=eval_fn, grad_clip=args.grad_clip,
                     progress_path=progress_path, budget=budget)
    ckpt = {
        "model_state": model.state_dict(),
        "info": info,
        "cfg_model": {"state_dim": cfg.state_dim, "act_dim": cfg.act_dim,
                      "max_ep_len": cfg.max_ep_len, "K_ctx": cfg.K_ctx,
                      "d_model": cfg.d_model, "dropout": cfg.dropout,
                      "action_tanh": cfg.action_tanh},
        "args": vars(args),
        "state_mean": state_mean, "state_std": state_std,
        "env": env_name, "dataset": args.dataset,
        "best_norm": tres["best_norm"], "total_train_s": tres["total_train_s"],
    }
    atomic_save(ckpt, out_dir / "ckpt.pt")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"info": info, "best_norm": tres["best_norm"],
                   "total_train_s": tres["total_train_s"],
                   "env": env_name, "dataset": args.dataset,
                   "n_params": n_params, "seed": int(args.seed),
                   "num_steps": int(args.num_steps),
                   "batch_size": int(args.batch_size),
                   "K_ctx": int(args.K_ctx)}, f, indent=2)
    print(f"[done] {out_dir}  best_norm={tres['best_norm']:.2f}", flush=True)
    env.close()


def cmd_eval(args):
    device = pick_device()
    set_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    info = ckpt["info"]
    cfg_model = ckpt["cfg_model"]
    env_name = ckpt["env"]
    cfg = RLModelConfig(**cfg_model)
    if info["kind"] == "transformer":
        a = ckpt["args"]
        model = build_transformer_model(cfg,
                                       n_layer=a.get("n_layer", info.get("n_layer", 3)),
                                       n_head=a.get("n_head", 1),
                                       activation=a.get("activation", "relu"))
    else:
        model = build_esssm_model(
            cfg, n_layer=ckpt["args"].get("n_layer", info.get("n_layer", 3)),
            K_max=info["K_max"], stoch_depth=0.0, cache_dir=args.cache_dir)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32)
    state_std = np.asarray(ckpt["state_std"], dtype=np.float32)
    cfg_env = ENV_CFG[env_name]
    env = make_env(env_name)
    out = {"env": env_name, "dataset": ckpt["dataset"], "info": info,
           "K_rt": args.K_rt}
    for tar in cfg_env["targets"]:
        rets, lens = rollout_many(
            env, cfg.state_dim, cfg.act_dim, model,
            max_ep_len=cfg.max_ep_len, scale=cfg_env["scale"],
            target_return_scaled=tar / cfg_env["scale"],
            state_mean=state_mean, state_std=state_std, device=device,
            num_episodes=args.num_eval_episodes, K_rt=args.K_rt)
        summary = summarize_rollouts(env_name, rets)
        summary["target_return"] = int(tar)
        summary["len_mean"] = float(lens.mean())
        out[f"t{int(tar)}"] = summary
    env.close()
    dummy = {
        "states": torch.zeros(args.bench_batch, cfg.K_ctx, cfg.state_dim),
        "actions": torch.zeros(args.bench_batch, cfg.K_ctx, cfg.act_dim),
        "rtg": torch.zeros(args.bench_batch, cfg.K_ctx, 1),
        "timesteps": torch.zeros(args.bench_batch, cfg.K_ctx, dtype=torch.long),
        "mask": torch.ones(args.bench_batch, cfg.K_ctx),
    }
    out["bench"] = bench_forward(model, dummy, K_rt=args.K_rt,
                                warmup=args.bench_warmup,
                                repeats=args.bench_repeats)
    out["n_params_ckpt"] = int(info.get("n_params", count_params(model)))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def cmd_export_compact(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    info = ckpt["info"]
    assert info["kind"] == "esssm", "source ckpt must be ES-SSM"
    cfg_model = ckpt["cfg_model"]
    cfg = RLModelConfig(**cfg_model)
    source = build_esssm_model(
        cfg, n_layer=ckpt["args"].get("n_layer", info.get("n_layer", 3)),
        K_max=info["K_max"], stoch_depth=0.0, cache_dir=args.cache_dir)
    source.load_state_dict(ckpt["model_state"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Ks = [int(k) for k in args.Ks.split(",")]
    manifest = {}
    for K in Ks:
        compact = build_compact_esssm_from_prefix(source, K=K,
                                                  cache_dir=args.cache_dir)
        n_params = count_params(compact)
        ck = {
            "model_state": compact.state_dict(),
            "info": {**info, "K_max": K, "n_params": n_params,
                     "source_K_max": info["K_max"]},
            "cfg_model": cfg_model, "args": ckpt["args"],
            "state_mean": ckpt["state_mean"], "state_std": ckpt["state_std"],
            "env": ckpt["env"], "dataset": ckpt["dataset"],
        }
        path = out_dir / f"esssm_K{K}.pt"
        atomic_save(ck, path)
        manifest[K] = {"path": str(path), "n_params": n_params}
        print(f"  K={K}: {n_params/1e6:.3f}M params -> {path}", flush=True)
        del compact
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _model_args_for_tag(tag: str) -> List[str]:
    table = {
        "dt":          ["--model", "dt"],
        "esssm_K32":   ["--model", "esssm", "--match-params-to", "dt", "--K-max", "32"],
        "tf_matchK2":  ["--model", "transformer", "--match-params-to", "esssm_K",
                        "--match-K", "2",  "--ref-d-essm", "72"],
        "tf_matchK4":  ["--model", "transformer", "--match-params-to", "esssm_K",
                        "--match-K", "4",  "--ref-d-essm", "72"],
        "tf_matchK8":  ["--model", "transformer", "--match-params-to", "esssm_K",
                        "--match-K", "8",  "--ref-d-essm", "72"],
        "tf_matchK16": ["--model", "transformer", "--match-params-to", "esssm_K",
                        "--match-K", "16", "--ref-d-essm", "72"],
    }
    if tag not in table:
        raise ValueError(f"unknown tag {tag!r}; expected one of {list(table)}")
    return table[tag]


def _run_dir(project_root: Path, tag: str, env: str, dataset: str, seed: int) -> Path:
    sub = f"{env}-{dataset}" if seed == 42 else f"{env}-{dataset}/seed{seed}"
    return project_root / "runs" / tag / sub


def _is_running(run_dir: Path, stale_s: int = 600) -> bool:
    p = run_dir / "progress.jsonl"
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < stale_s


def _visible_devices():
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count == 0:
        return ["-1"]
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [value.strip() for value in mask.split(",") if value.strip()] if mask is not None else [str(i) for i in range(count)]
    if len(devices) < count:
        raise RuntimeError("CUDA device count does not match the inherited visibility mask.")
    return devices[:count]


def cmd_sweep_train(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    tags = ALL_TAGS if args.only_tags is None else tuple(args.only_tags.split(","))
    project_root = Path(args.project_root)
    jobs = []
    for seed in seeds:
        for env, dataset, _ in DATASETS:
            for tag in tags:
                if args.only_env and env != args.only_env:
                    continue
                run_dir = _run_dir(project_root, tag, env, dataset, seed)
                if (run_dir / "summary.json").exists():
                    continue
                if _is_running(run_dir):
                    continue
                jobs.append((tag, env, dataset, seed, run_dir))
    print(f"[sweep-train] {len(jobs)} pending jobs over {len(tags)} tags x "
          f"{len(seeds)} seeds x 9 datasets")
    if args.dry_run:
        for tag, env, dataset, seed, run_dir in jobs:
            print(f"  {tag:14s} {env:12s} {dataset:14s} seed={seed} -> {run_dir}")
        return
    spg = max(1, args.slots_per_gpu)
    devices = _visible_devices()
    n_gpus = len(devices)
    n_slots = n_gpus * spg
    slot_to_gpu = [(i // spg) for i in range(n_slots)]
    slots: List[Optional[subprocess.Popen]] = [None] * n_slots
    slot_meta: List[Optional[tuple]] = [None] * n_slots
    print(f"[sweep-train] launching with {n_slots} slots ({spg}/GPU x {n_gpus} GPUs)")
    idx = 0
    start = time.time()
    failures = []

    def pick_slot():
        for i, p in enumerate(slots):
            if p is None:
                return i
        return None

    def poll():
        for i, p in enumerate(slots):
            if p is not None and p.poll() is not None:
                tag, env, dataset, seed = slot_meta[i]
                print(f"[slot{i} GPU{slot_to_gpu[i]}] finished "
                      f"{tag}/{env}-{dataset} seed={seed} rc={p.returncode}",
                      flush=True)
                if p.returncode != 0:
                    failures.append((slot_meta[i], p.returncode))
                slots[i] = None
                slot_meta[i] = None

    def spawn(tag, env, dataset, seed, run_dir, gpu_id):
        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = devices[gpu_id]
        env_vars.setdefault("MUJOCO_GL", "egl")
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "train",
               "--env", env, "--dataset", dataset,
               *_model_args_for_tag(tag),
               "--seed", str(seed),
               "--num-steps", str(args.num_steps),
               "--out-dir", str(run_dir),
               "--hdf5-dir", str(args.hdf5_dir),
               "--cache-dir", str(args.cache_dir)]
        log_path = run_dir / "train.log"
        with open(log_path, "ab") as fh:
            return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, close_fds=True,
                                    env=env_vars)

    while (idx < len(jobs) and not failures) or any(p is not None for p in slots):
        poll()
        while idx < len(jobs) and not failures and (s := pick_slot()) is not None:
            tag, env, dataset, seed, run_dir = jobs[idx]
            gpu = slot_to_gpu[s]
            print(f"[slot{s} GPU{gpu}] launch {tag}/{env}-{dataset} seed={seed}"
                  f"  ({idx+1}/{len(jobs)})  wall={int(time.time()-start)}s",
                  flush=True)
            slots[s] = spawn(tag, env, dataset, seed, run_dir, gpu)
            slot_meta[s] = (tag, env, dataset, seed)
            idx += 1
        time.sleep(15)
    print(f"[sweep-train] all done. wall={int(time.time()-start)}s")
    if failures:
        raise RuntimeError(f"{len(failures)} training jobs failed; {len(jobs) - idx} jobs were not launched. First failure: {failures[0]}")


def cmd_sweep_eval(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    project_root = Path(args.project_root)
    tasks = []
    for seed in seeds:
        for env, dataset, _ in DATASETS:
            for tag in ALL_TAGS:
                run_dir = _run_dir(project_root, tag, env, dataset, seed)
                ckpt = run_dir / "ckpt.pt"
                out_json = run_dir / "eval.json"
                if not ckpt.exists():
                    continue
                if out_json.exists() and not args.redo:
                    continue
                tasks.append((tag, env, dataset, seed, ckpt, out_json))
            esssm_run_dir = _run_dir(project_root, "esssm_K32", env, dataset, seed)
            manifest = esssm_run_dir / "compact" / "manifest.json"
            if manifest.exists():
                m = json.loads(manifest.read_text())
                for K_str, info in m.items():
                    K = int(K_str)
                    cpath = Path(info["path"])
                    out_json = esssm_run_dir / "compact" / f"eval_K{K}.json"
                    if out_json.exists() and not args.redo:
                        continue
                    tasks.append(("esssm_compact", env, dataset, seed, cpath, out_json))
    print(f"[sweep-eval] {len(tasks)} eval tasks pending")
    if args.dry_run:
        for t in tasks[:30]:
            print(f"  {t[0]:14s} {t[1]:12s} {t[2]:14s} seed={t[3]}  ->  {t[5]}")
        return
    spg = max(1, args.slots_per_gpu)
    devices = _visible_devices()
    n_gpus = len(devices)
    n_slots = n_gpus * spg
    slot_to_gpu = [(i // spg) for i in range(n_slots)]
    slots: List[Optional[subprocess.Popen]] = [None] * n_slots
    slot_meta: List[Optional[tuple]] = [None] * n_slots
    idx = 0
    start = time.time()
    failures = []

    def pick_slot():
        for i, p in enumerate(slots):
            if p is None:
                return i
        return None

    def poll():
        for i, p in enumerate(slots):
            if p is not None and p.poll() is not None:
                tag, env, dataset, seed, _, _ = slot_meta[i]
                print(f"[slot{i} GPU{slot_to_gpu[i]}] eval finished "
                      f"{tag}/{env}-{dataset} seed={seed} rc={p.returncode}",
                      flush=True)
                if p.returncode != 0:
                    failures.append((slot_meta[i], p.returncode))
                slots[i] = None
                slot_meta[i] = None

    def spawn(ckpt, out_json, gpu_id):
        env_vars = os.environ.copy()
        env_vars["CUDA_VISIBLE_DEVICES"] = devices[gpu_id]
        env_vars.setdefault("MUJOCO_GL", "egl")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "eval",
               "--ckpt", str(ckpt), "--out-json", str(out_json),
               "--num-eval-episodes", str(args.num_eval_episodes),
               "--bench-batch", "1", "--bench-warmup", "10",
               "--bench-repeats", "50",
               "--cache-dir", str(args.cache_dir)]
        log_path = out_json.with_suffix(".log")
        with open(log_path, "ab") as fh:
            return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, close_fds=True,
                                    env=env_vars)

    while (idx < len(tasks) and not failures) or any(p is not None for p in slots):
        poll()
        while idx < len(tasks) and not failures and (s := pick_slot()) is not None:
            tag, env, dataset, seed, ckpt, out_json = tasks[idx]
            gpu = slot_to_gpu[s]
            print(f"[slot{s} GPU{gpu}] eval {tag}/{env}-{dataset} seed={seed}"
                  f"  ({idx+1}/{len(tasks)})  wall={int(time.time()-start)}s",
                  flush=True)
            slots[s] = spawn(ckpt, out_json, gpu)
            slot_meta[s] = (tag, env, dataset, seed, ckpt, out_json)
            idx += 1
        time.sleep(10)
    print(f"[sweep-eval] all done. wall={int(time.time()-start)}s")
    if failures:
        raise RuntimeError(f"{len(failures)} evaluation jobs failed; {len(tasks) - idx} jobs were not launched. First failure: {failures[0]}")


def cmd_sweep_export(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    project_root = Path(args.project_root)
    Ks = args.Ks
    pending = []
    for seed in seeds:
        for env, dataset, _ in DATASETS:
            run_dir = _run_dir(project_root, "esssm_K32", env, dataset, seed)
            ckpt = run_dir / "ckpt.pt"
            out_dir = run_dir / "compact"
            if not ckpt.exists():
                continue
            if (out_dir / "manifest.json").exists() and not args.redo:
                continue
            pending.append((env, dataset, seed, ckpt, out_dir))
    print(f"[sweep-export] {len(pending)} cells to export")
    for env, dataset, seed, ckpt, out_dir in pending:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "export-compact",
               "--ckpt", str(ckpt), "--out-dir", str(out_dir),
               "--Ks", Ks, "--cache-dir", str(args.cache_dir)]
        print(f"  export {env}-{dataset} seed={seed}")
        subprocess.run(cmd, check=False)


def cmd_tabulate(args):
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for tag in ALL_TAGS:
        for seed in seeds:
            for env, dataset, _ in DATASETS:
                run_dir = _run_dir(project_root, tag, env, dataset, seed)
                summary_p = run_dir / "summary.json"
                eval_p = run_dir / "eval.json"
                if not summary_p.exists():
                    continue
                summary = json.loads(summary_p.read_text())
                info = summary.get("info", {})
                row = {
                    "tag": tag, "env": env, "dataset": dataset, "seed": seed,
                    "family": "Transformer" if "transformer" in info.get("kind", "")
                              else "ES-SSM",
                    "K_deploy": _deploy_K_of(tag, info),
                    "d_model": info.get("d"),
                    "params": summary.get("n_params"),
                    "best_norm_train": summary.get("best_norm"),
                    "train_s": summary.get("total_train_s"),
                }
                if eval_p.exists():
                    ev = json.loads(eval_p.read_text())
                    targets = [k for k in ev if k.startswith("t") and isinstance(ev[k], dict)]
                    if targets:
                        primary = ev[sorted(targets, key=lambda k: -int(k[1:]))[0]]
                        row["norm_mean"] = primary.get("norm_mean")
                        row["norm_std"] = primary.get("norm_std")
                        row["raw_mean"] = primary.get("raw_mean")
                        row["target_return"] = primary.get("target_return")
                    bench = ev.get("bench", {})
                    row["lat_ms"] = bench.get("lat_ms")
                    row["tput_ktok_s"] = bench.get("tput_ktok_s")
                    row["cuda_mb"] = bench.get("cuda_mb")
                rows.append(row)
    for env, dataset, _ in DATASETS:
        for seed in seeds:
            esssm_dir = _run_dir(project_root, "esssm_K32", env, dataset, seed)
            mf = esssm_dir / "compact" / "manifest.json"
            if not mf.exists():
                continue
            manifest = json.loads(mf.read_text())
            for K_str, info_k in manifest.items():
                K = int(K_str)
                eval_p = esssm_dir / "compact" / f"eval_K{K}.json"
                if not eval_p.exists():
                    continue
                ev = json.loads(eval_p.read_text())
                targets = [k for k in ev if k.startswith("t") and isinstance(ev[k], dict)]
                primary = ev[sorted(targets, key=lambda k: -int(k[1:]))[0]] if targets else {}
                bench = ev.get("bench", {})
                rows.append({
                    "tag": f"esssm_compact_K{K}", "env": env, "dataset": dataset,
                    "seed": seed, "family": "ES-SSM_compact", "K_deploy": K,
                    "params": info_k.get("n_params"),
                    "norm_mean": primary.get("norm_mean"),
                    "norm_std": primary.get("norm_std"),
                    "raw_mean": primary.get("raw_mean"),
                    "target_return": primary.get("target_return"),
                    "lat_ms": bench.get("lat_ms"),
                    "tput_ktok_s": bench.get("tput_ktok_s"),
                    "cuda_mb": bench.get("cuda_mb"),
                })
    _write_csv(out_dir / "per_seed.csv", rows)
    per_env = _aggregate_by_env(rows)
    _write_csv(out_dir / "per_env_mean.csv", per_env)
    by_K = _aggregate_by_K(per_env)
    _write_csv(out_dir / "by_K.csv", by_K)
    print(f"[tabulate] wrote {out_dir}/per_seed.csv ({len(rows)} rows), "
          f"per_env_mean.csv ({len(per_env)} rows), by_K.csv ({len(by_K)} rows)")


def _deploy_K_of(tag: str, info: dict) -> Optional[int]:
    if tag in ("dt", "tf_matchK32"):
        return 32
    if tag == "esssm_K32":
        return 32
    if tag.startswith("tf_matchK"):
        return int(tag.split("K")[-1])
    return None


def _aggregate_by_env(rows):
    groups: Dict[Tuple, List[dict]] = {}
    for r in rows:
        key = (r["tag"], r["env"], r["dataset"])
        groups.setdefault(key, []).append(r)
    out = []
    for (tag, env, ds), rs in groups.items():
        keep = {"tag": tag, "env": env, "dataset": ds,
                "family": rs[0].get("family"),
                "K_deploy": rs[0].get("K_deploy"),
                "d_model": rs[0].get("d_model"),
                "params": rs[0].get("params"),
                "n_seeds": len(rs)}
        for col in ("best_norm_train", "norm_mean", "raw_mean",
                    "lat_ms", "tput_ktok_s", "cuda_mb", "train_s"):
            xs = [r.get(col) for r in rs if r.get(col) is not None]
            if xs:
                m = sum(xs) / len(xs)
                v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1) if len(xs) > 1 else 0.0
                keep[f"{col}_mean"] = m
                keep[f"{col}_std"] = math.sqrt(v)
            else:
                keep[f"{col}_mean"] = None
                keep[f"{col}_std"] = None
        out.append(keep)
    return out


def _aggregate_by_K(per_env_rows):
    groups: Dict[Tuple, List[dict]] = {}
    for r in per_env_rows:
        K = r.get("K_deploy")
        if K is None:
            continue
        key = (r["tag"], r.get("family"), K)
        groups.setdefault(key, []).append(r)
    out = []
    for (tag, family, K), rs in groups.items():
        xs = [r.get("norm_mean_mean") for r in rs if r.get("norm_mean_mean") is not None]
        m = sum(xs) / len(xs) if xs else None
        v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1) if xs and len(xs) > 1 else 0.0
        out.append({
            "tag": tag, "family": family, "K": K, "n_envs": len(rs),
            "params": rs[0].get("params"),
            "norm_mean_over_envs": m,
            "norm_std_over_envs": math.sqrt(v) if m is not None else None,
            "best_norm_train_mean": _safe_mean(r.get("best_norm_train_mean") for r in rs),
            "lat_ms_mean": _safe_mean(r.get("lat_ms_mean") for r in rs),
            "tput_ktok_s_mean": _safe_mean(r.get("tput_ktok_s_mean") for r in rs),
            "cuda_mb_mean": _safe_mean(r.get("cuda_mb_mean") for r in rs),
            "train_s_mean": _safe_mean(r.get("train_s_mean") for r in rs),
        })
    return out


def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _write_csv(path: Path, rows):
    if not rows:
        print(f"[skip] no rows for {path}")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"[wrote] {path} ({len(rows)} rows)")


def _add_train_args(p: argparse.ArgumentParser):
    p.add_argument("--env", choices=list(ENV_CFG.keys()), required=True)
    p.add_argument("--dataset",
                  choices=["medium", "medium-replay", "medium-expert"], required=True)
    p.add_argument("--model", choices=["dt", "transformer", "esssm"], required=True)
    p.add_argument("--match-params-to", dest="match_params_to",
                  choices=[None, "dt", "esssm_K"], default=None)
    p.add_argument("--match-K", dest="match_K", type=int, default=32)
    p.add_argument("--ref-d-essm", dest="ref_d_essm", type=int, default=72)
    p.add_argument("--ref-n-layer-essm", dest="ref_n_layer_essm", type=int, default=3)
    p.add_argument("--K-max", dest="K_max", type=int, default=DEFAULT_HYPER["K_max"])
    p.add_argument("--K-ctx", dest="K_ctx", type=int, default=DEFAULT_HYPER["K_ctx"])
    p.add_argument("--embed-dim", dest="embed_dim", type=int,
                  default=DEFAULT_HYPER["embed_dim"])
    p.add_argument("--n-layer", dest="n_layer", type=int,
                  default=DEFAULT_HYPER["n_layer"])
    p.add_argument("--n-head", dest="n_head", type=int,
                  default=DEFAULT_HYPER["n_head"])
    p.add_argument("--activation", default=DEFAULT_HYPER["activation"])
    p.add_argument("--dropout", type=float, default=DEFAULT_HYPER["dropout"])
    p.add_argument("--stoch-depth", dest="stoch_depth", type=float, default=0.0)
    p.add_argument("--batch-size", dest="batch_size", type=int,
                  default=DEFAULT_HYPER["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULT_HYPER["lr"])
    p.add_argument("--weight-decay", dest="weight_decay", type=float,
                  default=DEFAULT_HYPER["weight_decay"])
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int,
                  default=DEFAULT_HYPER["warmup_steps"])
    p.add_argument("--num-steps", dest="num_steps", type=int,
                  default=DEFAULT_HYPER["num_steps"])
    p.add_argument("--eval-every", dest="eval_every", type=int, default=5000)
    p.add_argument("--num-eval-episodes-train", dest="num_eval_episodes_train",
                  type=int, default=DEFAULT_HYPER["num_eval_episodes_train"])
    p.add_argument("--grad-clip", dest="grad_clip", type=float,
                  default=DEFAULT_HYPER["grad_clip"])
    p.add_argument("--max-ep-len", dest="max_ep_len", type=int,
                  default=DEFAULT_HYPER["max_ep_len"])
    p.add_argument("--seed", type=int, default=DEFAULT_HYPER["seed"])
    p.add_argument("--out-dir", dest="out_dir", required=True)
    p.add_argument("--hdf5-dir", dest="hdf5_dir", default="./data")
    p.add_argument("--cache-dir", dest="cache_dir", default="./cache_filters")
    p.add_argument("--budget-enabled", dest="budget_enabled", action="store_true",
                  default=True)
    p.add_argument("--no-budget", dest="budget_enabled", action="store_false")
    p.add_argument("--budget-warmup-frac", dest="budget_warmup_frac", type=float,
                  default=DEFAULT_HYPER["budget_warmup_frac"])
    p.add_argument("--budget-full-every", dest="budget_full_every", type=int,
                  default=DEFAULT_HYPER["budget_full_every"])
    p.add_argument("--budget-K-min", dest="budget_K_min", type=int,
                  default=DEFAULT_HYPER["budget_K_min"])
    p.add_argument("--budget-ks", dest="budget_ks",
                  default=",".join(str(k) for k in DEFAULT_HYPER["budget_ks"]))


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    dl = sub.add_parser("download-data",
                        help="Download the 9 D4RL v2 HDF5 files.")
    dl.add_argument("--hdf5-dir", default="./data")
    dl.set_defaults(func=cmd_download_data)

    t = sub.add_parser("train", help="Train one (env, dataset, model, seed) cell.")
    _add_train_args(t)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval",
                       help="Run rollout + bench on a trained checkpoint.")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out-json", dest="out_json", required=True)
    e.add_argument("--K-rt", dest="K_rt", type=int, default=None)
    e.add_argument("--num-eval-episodes", dest="num_eval_episodes", type=int, default=100)
    e.add_argument("--bench-batch", dest="bench_batch", type=int, default=1)
    e.add_argument("--bench-warmup", dest="bench_warmup", type=int, default=10)
    e.add_argument("--bench-repeats", dest="bench_repeats", type=int, default=50)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--cache-dir", dest="cache_dir", default="./cache_filters")
    e.set_defaults(func=cmd_eval)

    x = sub.add_parser("export-compact",
                       help="Materialize independent compact ES-SSM(K) checkpoints.")
    x.add_argument("--ckpt", required=True)
    x.add_argument("--out-dir", dest="out_dir", required=True)
    x.add_argument("--Ks", default="2,4,8,16,32")
    x.add_argument("--cache-dir", dest="cache_dir", default="./cache_filters")
    x.set_defaults(func=cmd_export_compact)

    st = sub.add_parser("sweep-train",
                        help="Multi-GPU parallel training over the full grid.")
    st.add_argument("--project-root", dest="project_root", default="./project")
    st.add_argument("--hdf5-dir", dest="hdf5_dir", default="./data")
    st.add_argument("--cache-dir", dest="cache_dir", default="./cache_filters")
    st.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    st.add_argument("--only-tags", dest="only_tags", default=None)
    st.add_argument("--only-env", dest="only_env", default=None)
    st.add_argument("--num-steps", dest="num_steps", type=int,
                    default=DEFAULT_HYPER["num_steps"])
    st.add_argument("--slots-per-gpu", dest="slots_per_gpu", type=int, default=3)
    st.add_argument("--dry-run", dest="dry_run", action="store_true")
    st.set_defaults(func=cmd_sweep_train)

    se = sub.add_parser("sweep-eval",
                        help="Multi-GPU parallel eval over every trained ckpt.")
    se.add_argument("--project-root", dest="project_root", default="./project")
    se.add_argument("--cache-dir", dest="cache_dir", default="./cache_filters")
    se.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    se.add_argument("--num-eval-episodes", dest="num_eval_episodes", type=int, default=100)
    se.add_argument("--slots-per-gpu", dest="slots_per_gpu", type=int, default=3)
    se.add_argument("--redo", action="store_true")
    se.add_argument("--dry-run", dest="dry_run", action="store_true")
    se.set_defaults(func=cmd_sweep_eval)

    sx = sub.add_parser("sweep-export",
                        help="Compact-K export for every trained ES-SSM cell.")
    sx.add_argument("--project-root", dest="project_root", default="./project")
    sx.add_argument("--cache-dir", dest="cache_dir", default="./cache_filters")
    sx.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    sx.add_argument("--Ks", default="2,4,8,16,32")
    sx.add_argument("--redo", action="store_true")
    sx.set_defaults(func=cmd_sweep_export)

    tb = sub.add_parser("tabulate",
                        help="Aggregate eval JSONs into per-seed/per-env/by-K CSVs.")
    tb.add_argument("--project-root", dest="project_root", default="./project")
    tb.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    tb.add_argument("--out-dir", dest="out_dir", default="./results")
    tb.set_defaults(func=cmd_tabulate)

    return p


def main():
    p = build_argparser()
    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
