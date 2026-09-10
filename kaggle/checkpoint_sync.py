#!/usr/bin/env python3
import argparse
import glob
import os
import shutil


def push_hf(local_dir: str, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    latest = _find_latest(local_dir)
    if latest is None:
        print(f"[WARNING] No checkpoints in {local_dir} - nothing to send.")
        return
    print(f"[PUSH] {latest} -> hf://{repo_id}")
    api.upload_file(
        path_or_fileobj=latest,
        path_in_repo="latest.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[OK] Checkpoint sent as 'latest.pt' to {repo_id}")


def pull_hf(local_dir: str, repo_id: str):
    from huggingface_hub import hf_hub_download

    os.makedirs(local_dir, exist_ok=True)
    print(f"[PULL] hf://{repo_id}/latest.pt -> {local_dir}")
    try:
        path = hf_hub_download(repo_id=repo_id, filename="latest.pt", repo_type="model")
    except Exception as ex:
        print(f"[INFO] No checkpoint in {repo_id} (first session?): {ex}")
        return None
    dst = os.path.join(local_dir, "final.pt")
    shutil.copy(path, dst)
    print(f"[OK] Checkpoint ready: {dst} (use --resume_from {dst})")
    return dst


def push_kaggle(local_dir: str, dataset_slug: str):
    meta_path = os.path.join(local_dir, "dataset-metadata.json")
    if not os.path.exists(meta_path):
        os.system(f"kaggle datasets init -p {local_dir}")
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        meta["title"] = dataset_slug.split("/")[-1]
        meta["id"] = dataset_slug
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(
            f"[INFO] Created {meta_path} - the first push creates a new dataset ({dataset_slug}). You then MUST manually add it as 'Input' to the notebook on Kaggle (Add Data -> Your Datasets)."
        )
        os.system(f"kaggle datasets create -p {local_dir}")
    else:
        os.system(
            f'kaggle datasets version -p {local_dir} -m "checkpoint update" --dir-mode zip'
        )
    print(f"[OK] Checkpoint(s) sent to Kaggle Dataset: {dataset_slug}")


def pull_kaggle(local_dir: str, dataset_slug: str):
    input_guess = f"/kaggle/input/{dataset_slug.split('/')[-1]}"
    if not os.path.isdir(input_guess):
        print(
            f"[ERROR] {input_guess} not found. Add the dataset '{dataset_slug}' as Input to the notebook (Add Data -> Your Datasets) and run again."
        )
        return None
    latest = _find_latest(input_guess)
    if latest is None:
        print(f"[INFO] No checkpoints in {input_guess} (first session?).")
        return None
    os.makedirs(local_dir, exist_ok=True)
    dst = os.path.join(local_dir, "final.pt")
    shutil.copy(latest, dst)
    print(f"[OK] Checkpoint ready: {dst} (use --resume_from {dst})")
    return dst


def _find_latest(directory: str):
    candidates = glob.glob(os.path.join(directory, "final.pt"))
    if candidates:
        return candidates[0]
    step_files = glob.glob(os.path.join(directory, "step_*.pt"))
    if not step_files:
        return None

    def step_num(p):
        try:
            return int(os.path.basename(p).replace("step_", "").replace(".pt", ""))
        except ValueError:
            return -1

    return max(step_files, key=step_num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "kaggle"], default="hf")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--local_dir", default="checkpoints")
    ap.add_argument(
        "--repo_id", default=None, help="e.g. your-user/san2-checkpoints (backend hf)"
    )
    ap.add_argument(
        "--dataset_slug",
        default=None,
        help="e.g. your-user/san2-checkpoints (backend kaggle)",
    )
    args = ap.parse_args()
    if args.push == args.pull:
        raise SystemExit("Provide exactly one of: --push or --pull")
    if args.backend == "hf":
        if not args.repo_id:
            raise SystemExit("--backend hf requires --repo_id")
        (push_hf if args.push else pull_hf)(args.local_dir, args.repo_id)
    else:
        if not args.dataset_slug:
            raise SystemExit("--backend kaggle requires --dataset_slug")
        (push_kaggle if args.push else pull_kaggle)(args.local_dir, args.dataset_slug)


if __name__ == "__main__":
    main()
