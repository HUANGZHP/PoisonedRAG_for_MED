# Purpose: Implements download_medcpt_models.py in the PoisonedRAG project.

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download MedCPT models to local path")
    parser.add_argument("--output-root", type=str, default="models")
    parser.add_argument("--include-article-encoder", action="store_true")
    parser.add_argument("--hf-token", type=str, default="")
    return parser.parse_args()


def download_repo(repo_id: str, local_dir: Path, token: str):
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token or None,
        resume_download=True,
    )
    print(f"Downloaded {repo_id} -> {local_dir}")


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()

    query_dir = output_root / "ncbi" / "MedCPT-Query-Encoder"
    download_repo("ncbi/MedCPT-Query-Encoder", query_dir, args.hf_token)

    if args.include_article_encoder:
        article_dir = output_root / "ncbi" / "MedCPT-Article-Encoder"
        download_repo("ncbi/MedCPT-Article-Encoder", article_dir, args.hf_token)

    print("\nSet this before running retrieval:")
    print(f"export MEDCPT_QUERY_ENCODER_PATH={query_dir}")


if __name__ == "__main__":
    main()
