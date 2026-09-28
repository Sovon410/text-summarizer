"""Lightweight syntax/import-independent project smoke checks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

required = [
    ROOT / "frontend/index.html",
    ROOT / "frontend/script.js",
    ROOT / "frontend/style.css",
    ROOT / "backend/app/main.py",
    ROOT / "backend/app/summarizer.py",
    ROOT / "backend/requirements.txt",
    ROOT / "netlify.toml",
    ROOT / "render.yaml",
]

for path in required:
    assert path.exists(), f"Missing: {path}"

html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
js = (ROOT / "frontend/script.js").read_text(encoding="utf-8")
assert 'src="./script.js" defer' in html
assert 'id="text-input"' in html
assert 'getElementById("text-input")' in js
assert '.value.trim()' in js

backend = (ROOT / "backend/app/summarizer.py").read_text(encoding="utf-8")
assert "num_beams=NUM_BEAMS" in backend
assert "value.to(DEVICE)" in backend
assert "torch.inference_mode()" in backend

print("Smoke test passed.")
