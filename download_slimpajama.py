import argparse
import json
import os
import sys
import time
from pathlib import Path
from runtime_utils import file_sha256

REPO = "gmongaras/SlimPajama-627B_Reupload"


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out-dir", required=True)
    pa.add_argument("--train-files", type=int, default=1)
    pa.add_argument("--val-files", type=int, default=1)
    pa.add_argument("--test-files", type=int, default=1)
    pa.add_argument("--disable-xet", action="store_true", default=True)
    pa.add_argument("--revision", default="main")
    args = pa.parse_args()
    counts = {"train": args.train_files, "validation": args.val_files, "test": args.test_files}
    limits = {"train": 250, "validation": 30, "test": 30}
    if any(n < 0 or n > limits[k] for k, n in counts.items()) or not any(counts.values()):
        pa.error("Shard counts must be within the repository limits, with at least one requested file.")

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    revision = HfApi().repo_info(REPO, repo_type="dataset", revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve an immutable dataset revision.")
    print(f"dataset={REPO} revision={revision} requested_shards={counts}")
    if counts != limits:
        print("This is a subset download, not the full dataset.")

    to_fetch = []
    for i in range(args.train_files):
        to_fetch.append(f"data/train-{i:05d}-of-00250.parquet")
    for i in range(args.val_files):
        to_fetch.append(f"data/validation-{i:05d}-of-00030.parquet")
    for i in range(args.test_files):
        to_fetch.append(f"data/test-{i:05d}-of-00030.parquet")

    total_start = time.time()
    files, failures = [], []
    for f in to_fetch:
        t0 = time.time()
        try:
            path = hf_hub_download(
                REPO, f, repo_type="dataset",
                local_dir=str(out_dir),
                revision=revision,
            )
        except Exception as e:
            print(f"[fail] {f}: {e}", file=sys.stderr)
            failures.append({"file": f, "error": str(e)})
            continue
        sz = os.path.getsize(path) / 1e9
        files.append({"file": f, "sha256": file_sha256(path), "bytes": os.path.getsize(path)})
        print(f"[ok]   {f}  {sz:.2f} GB in {time.time()-t0:.1f}s  -> {path}")

    manifest = {"repository": REPO, "revision": revision, "requested_revision": args.revision,
                "requested_shards": counts, "files": files, "failures": failures, "complete": not failures}
    (out_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"total time: {time.time()-total_start:.1f}s")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
