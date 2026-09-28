"""Upload a locally saved fine-tuned T5 model to the Hugging Face Hub."""

import argparse
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Path to the saved model directory")
    parser.add_argument("repo_id", help="Hugging Face repo, e.g. Sovon410/text-summarizer-t5-small")
    parser.add_argument("--private", action="store_true", help="Create a private model repository")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    tokenizer.push_to_hub(args.repo_id, private=args.private)
    model.push_to_hub(args.repo_id, private=args.private)
    print(f"Uploaded model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
