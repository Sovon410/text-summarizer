import os
import re
from functools import lru_cache


MODEL_ID = os.getenv("MODEL_ID", "./model")
INPUT_MAX_LENGTH = int(os.getenv("INPUT_MAX_LENGTH", "512"))
SUMMARY_MAX_LENGTH = int(os.getenv("SUMMARY_MAX_LENGTH", "150"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))


def clean_text(text: str) -> str:
    """Normalize whitespace and remove HTML-like tags used in raw text."""
    text = re.sub(r"\r\n?", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    return text.strip().lower()


@lru_cache(maxsize=1)
def load_model():
    """
    Load PyTorch, tokenizer, and model only when summarization
    is actually requested.

    This keeps FastAPI startup lightweight so Render can detect
    the service port quickly.
    """

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Render's web service does not require CUDA.
    # Use CPU unless CUDA is explicitly enabled.
    use_cuda = os.getenv("USE_CUDA", "false").lower() == "true"

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def summarize_text(dialogue: str) -> str:
    cleaned = clean_text(dialogue)

    if not cleaned:
        raise ValueError("Dialogue cannot be empty after cleaning.")

    tokenizer, model, device = load_model()

    # The prefix matches the summarization task used during training.
    prompt = f"summarize: {cleaned}"

    inputs = tokenizer(
        prompt,
        max_length=INPUT_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch_inference_mode():
        generated_ids = model.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            max_length=SUMMARY_MAX_LENGTH,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    summary = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

    if not summary:
        raise ValueError("The model returned an empty summary.")

    return summary


def torch_inference_mode():
    """
    Import torch only when inference is actually needed.
    """
    import torch

    return torch.inference_mode()