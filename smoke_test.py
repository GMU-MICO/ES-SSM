import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import runtime_utils as runtime


ROOT = Path(__file__).resolve().parent
MODULES = ["essm_LM", "transformer_pp_LM", "gla_LM"]


def load_module(name, root=ROOT, tag="smoke"):
    spec = importlib.util.spec_from_file_location(tag + "_" + name, Path(root) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(module, folder, device="cpu"):
    module.DEVICE = torch.device(device)
    module.AMP_DEVICE = device
    module.AMP_DTYPE = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cpu":
        module.autocast = contextlib.nullcontext
    module.CFG.update(
        seed=17, vocab_size=258, seq_len=16, d_model=16, n_layers=2,
        K_max=4, K_chunk=2, batch_size=1, grad_accum_steps=1,
        grad_ckpt=False, dropout=0., stoch_depth=0., num_workers=0,
        target_tokens=32, tokens_between_val=16, eval_batches=2, test_batches=2,
        val_segments=3, test_segments=3, eval_k=[2, 4], chunk_size=4,
        budget_ks=[2, 4], budget_warmup_tokens=0, budget_full_every=2,
        filter_cache_dir=str(Path(folder) / "filters"),
        dataset_dir=str(Path(folder) / "dataset"), output_dir=str(Path(folder) / "output"),
    )
    if hasattr(module, "SIZES"):
        module.SIZES = [
            dict(K=k, d_model=16, n_layers=1, n_heads=2, d_ffn=32, head_kv=8, head_v=8)
            for k in [2, 4]
        ]


def model_for(module, name):
    if name == "essm_LM":
        return module.ESSSMLanguageModel(module.CFG["filter_cache_dir"])
    if name == "transformer_pp_LM":
        return module.TransformerLM(module.SIZES[0])
    if name == "gla_LM":
        return module.GLALM(module.SIZES[0])
    raise ValueError(f"Unknown model: {name}")


def gla_recurrence(q, k, v, g):
    state = torch.zeros(*q.shape[:2], q.shape[-1], v.shape[-1], device=q.device, dtype=q.dtype)
    outputs = []
    for t in range(q.shape[2]):
        state = g[:, :, t].exp().unsqueeze(-1) * state + k[:, :, t, :, None] * v[:, :, t, None, :]
        outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, :, t], state))
    return torch.stack(outputs, dim=2)


def check_gla(device="cpu"):
    module = load_module("gla_LM")
    ones = torch.ones(1, 1, 2, 1, device=device)
    gates = torch.full_like(ones, math.log(0.5))
    expected = torch.tensor([1., 1.5], device=device)
    for chunk in [1, 2]:
        assert torch.allclose(module.gla_chunkwise(ones, ones, ones, gates, chunk).flatten(), expected)
    torch.manual_seed(71)
    values = [torch.randn(2, 2, 12, n, device=device, dtype=torch.float64) for n in [3, 3, 5]]
    values.append(-torch.rand(2, 2, 12, 3, device=device, dtype=torch.float64))
    for chunk in [1, 2, 3, 4, 6, 12]:
        a = [x.clone().requires_grad_() for x in values]
        b = [x.clone().requires_grad_() for x in values]
        reference = gla_recurrence(*a)
        actual = module.gla_chunkwise(*b, chunk)
        torch.testing.assert_close(reference, actual, rtol=1e-10, atol=1e-10)
        ga = torch.autograd.grad(reference.square().sum(), a)
        gb = torch.autograd.grad(actual.square().sum(), b)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-9)
    return {"two_step_oracle": [1.0, 1.5], "chunk_sizes": [1, 2, 3, 4, 6, 12], "gradients": "PASS"}


def check_runtime(folder):
    length = 12
    i = torch.arange(length).float()
    s = i[:, None] + i[None, :] + 2
    h = 2 / (s ** 3 - s + 1e-6)
    x = torch.randn(length, 3)
    torch.testing.assert_close(runtime.hmv(torch.fft.rfft(runtime.hankel_kernel(length)), x), h @ x, rtol=1e-5, atol=1e-6)
    state = torch.get_rng_state().clone()
    sigma, phi = runtime.hankel_basis(Path(folder) / "hankel", length, 3)
    assert torch.equal(state, torch.get_rng_state())
    sigma2, phi2 = runtime.hankel_basis(Path(folder) / "hankel", length, 3)
    assert torch.equal(state, torch.get_rng_state()) and torch.equal(sigma, sigma2) and torch.equal(phi, phi2)
    torch.testing.assert_close(phi.T @ phi, torch.eye(3), atol=1e-5, rtol=1e-5)
    assert float((h @ phi - phi * sigma).norm()) < 1e-5
    data = Path(folder) / "bytes" / "data"
    data.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["abcd", "éZ", "", None, "0123456789"]}), data / "validation-0.parquet", compression=None)
    cache = Path(folder) / "cache"
    t = runtime.build_eval_cache(data.parent, cache, "validation", 3, 7, 5)
    expected = torch.tensor([[99, 100, 101, 102, 12, 197], [197, 171, 92, 12, 50, 51], [51, 52, 53, 54, 55, 56]], dtype=torch.int16)
    assert torch.equal(t, expected)
    assert torch.equal(t, runtime.build_eval_cache(data.parent, cache, "validation", 3, 7, 5))
    try:
        runtime.build_eval_cache(data.parent, cache, "validation", 3, 8, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched cache seed was accepted.")
    return {"hankel": "PASS", "cold_warm_rng": "PASS", "byte_oracle": "PASS", "cache_provenance": "PASS"}


def check_models(folder, device="cpu"):
    output = []
    tokens = (torch.arange(32, device=device).reshape(2, 16) + 2).long()
    for name in MODULES:
        module = load_module(name)
        configure(module, Path(folder) / name, device)
        torch.manual_seed(191)
        model = model_for(module, name).to(device)
        budgets = [2, 4] if name == "essm_LM" else [None]
        for checkpointing in [False, True]:
            model.use_ckpt = checkpointing
            for k in budgets:
                model.train()
                model.zero_grad(set_to_none=True)
                with module.autocast():
                    logits = model(tokens) if k is None else model(tokens, K_rt=k)
                    loss = logits.float().square().mean()
                assert logits.shape == (2, 16, 258) and bool(torch.isfinite(logits).all())
                loss.backward()
                assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                output.append({"model": name, "K": k, "checkpointing": checkpointing, "finite_forward_backward": True})
        saved = Path(folder) / (name + ".pt")
        runtime.atomic_save({"model": model.state_dict()}, saved)
        model.load_state_dict(torch.load(saved, map_location=device, weights_only=True)["model"], strict=True)
    return output


def check_lm_compact(folder):
    module = load_module("essm_LM", tag="compact")
    configure(module, Path(folder) / "compact")
    torch.manual_seed(101)
    source = module.ESSSMLanguageModel(module.CFG["filter_cache_dir"]).eval()
    tokens = (torch.arange(32).reshape(2, 16) + 2).long()
    sizes = {}
    with torch.no_grad():
        changed = tokens.clone()
        changed[:, 9:] = 157
        torch.testing.assert_close(source(tokens)[:, :9], source(changed)[:, :9])
        for k in [2, 4]:
            expected = source(tokens, K_rt=k)
            compact = module.build_compact(module.CFG["filter_cache_dir"], source, k).eval()
            path = Path(folder) / f"compact_K{k}.pt"
            runtime.atomic_save({"model": compact.state_dict()}, path)
            original_k = module.CFG["K_max"]
            try:
                module.CFG["K_max"] = k
                loaded = module.ESSSMLanguageModel(str(Path(folder) / f"fresh_{k}")).eval()
            finally:
                module.CFG["K_max"] = original_k
            loaded.load_state_dict(torch.load(path, weights_only=True)["model"], strict=True)
            assert all(layer.M.shape[0] == k for layer in loaded.layers)
            torch.testing.assert_close(loaded(tokens), expected, rtol=1e-5, atol=1e-6)
            sizes[k] = (module.count_params(loaded), path.stat().st_size)
    assert sizes[2][0] < sizes[4][0] and sizes[2][1] < sizes[4][1]
    return {"causality": "PASS", "compact_reload": "PASS", "budget_equivalence": "PASS"}


def check_rl(folder):
    import d4rl_mujoco_RL as module

    cfg = module.RLModelConfig(state_dim=11, act_dim=3, max_ep_len=16, K_ctx=4,
                               d_model=16, dropout=0.)
    source = module.build_esssm_model(cfg, n_layer=1, K_max=4,
                                     cache_dir=str(Path(folder) / "rl_filters")).eval()
    batch = {"states": torch.randn(2, 4, 11), "actions": torch.randn(2, 4, 3),
             "rtg": torch.randn(2, 4, 1), "timesteps": torch.arange(4).expand(2, -1),
             "mask": torch.ones(2, 4)}
    for k in [2, 4]:
        with torch.no_grad():
            expected = source(**batch, K_rt=k)
        compact = module.build_compact_esssm_from_prefix(source, k, str(Path(folder) / "rl_compact"))
        saved = Path(folder) / f"rl_K{k}.pt"
        runtime.atomic_save({"model_state": compact.state_dict()}, saved)
        loaded = module.build_esssm_model(cfg, n_layer=1, K_max=k,
                                         cache_dir=str(Path(folder) / f"rl_fresh_{k}")).eval()
        loaded.load_state_dict(torch.load(saved, weights_only=True)["model_state"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(loaded(**batch), expected, rtol=1e-5, atol=1e-6)
        loaded.train()
        loss = loaded(**batch).square().mean()
        loss.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in loaded.parameters() if p.grad is not None)
    with torch.no_grad():
        altered = dict(batch, actions=batch["actions"].clone(), states=batch["states"].clone())
        altered["actions"][:, 2:] += 9
        altered["states"][:, 2:] += 9
        torch.testing.assert_close(source(**batch)[:, :2], source(**altered)[:, :2])
        same_action_time = dict(batch, actions=batch["actions"] + 9)
        torch.testing.assert_close(source(**batch)[:, :1], source(**same_action_time)[:, :1])
    return {"compact_equivalence": "PASS", "finite_forward_backward": "PASS", "causality": "PASS"}


def check_training(folder, device="cpu"):
    data = Path(folder) / "training_data" / "data"
    data.mkdir(parents=True)
    for split in ["train", "validation", "test"]:
        pq.write_table(pa.table({"text": ["A small deterministic training fixture.\n" * 12]}),
                       data / f"{split}-0.parquet")
    original_argv = list(sys.argv)
    reports = []
    try:
        for name in MODULES:
            module = load_module(name, tag="training")
            output = Path(folder) / f"train_{name}"
            configure(module, output, device)
            common = ["--data-dir", str(data.parent), "--output-dir", str(output), "--num-workers", "0"]
            if name == "essm_LM":
                common += ["--filter-cache-dir", str(output / "filters")]
            with contextlib.redirect_stdout(io.StringIO()):
                sys.argv = [name, *common]
                module.main()
                sys.argv = [name, *common, "--inference-only"]
                module.main()
            checkpoints = list((output / "checkpoints").glob("*.pt"))
            assert checkpoints
            reports.append({"model": name, "tiny_training_and_inference": "PASS"})
    finally:
        sys.argv = original_argv
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available.")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="esssm-smoke-") as folder:
        report = {"runtime": check_runtime(folder), "gla": check_gla(args.device),
                  "models": check_models(folder, args.device), "lm_compact": check_lm_compact(folder),
                  "offline_rl": check_rl(folder), "training": check_training(folder, args.device),
                  "scope": "synthetic functional checks"}
    print(json.dumps({"status": "PASS", **report}, indent=2))


if __name__ == "__main__":
    main()
