# Text Summarizer

An end-to-end text summarization web application using a fine-tuned T5 model, FastAPI, HTML/CSS/JavaScript, Netlify, and a separate Python model API.

## Architecture

```text
Browser
   |
   v
Netlify (static frontend)
   |
   | /api/summarize
   v
FastAPI model API (Render)
   |
   v
Fine-tuned T5 model
```

Netlify hosts the UI. The Python inference service runs separately because the current Netlify Functions getting-started path is for JavaScript/TypeScript and Go, while Python is supported in Netlify's build environment rather than as the FastAPI runtime used here.

## Project structure

```text
text-summarizer/
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── summarizer.py
│   ├── .env.example
│   ├── .python-version
│   └── requirements.txt
├── training/
│   └── textSummarizer_fixed.ipynb
├── data/
│   └── README.md
├── scripts/
│   └── upload_model_to_huggingface.py
├── netlify.toml
├── render.yaml
└── .gitignore
```

## 1. Train/save the model locally

Use the fixed notebook in `training/textSummarizer_fixed.ipynb`.

The notebook expects the three SamSum CSVs in `data/` and saves the trained model to `training/artifacts/saved_summary_model`.

## 2. Upload the trained model to Hugging Face

Install the training dependencies, log in to Hugging Face, then run:

```bash
python scripts/upload_model_to_huggingface.py training/artifacts/saved_summary_model YOUR_USERNAME/text-summarizer-t5-small
```

If you use a private model repository, add `--private`. For a private model, configure the Hugging Face token in the Render service and update `backend/app/summarizer.py` to pass that token to `from_pretrained`.

## 3. Deploy the FastAPI API to Render

Create a Render Web Service from this repository.

Recommended settings:

- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`
- Python: 3.12.x

Set these environment variables in Render:

```text
MODEL_ID=YOUR_USERNAME/text-summarizer-t5-small
ALLOWED_ORIGINS=https://YOUR-NETLIFY-SITE.netlify.app
```

Render's FastAPI deployment documentation uses the same `pip install -r requirements.txt` and `uvicorn ... --host 0.0.0.0 --port $PORT` pattern.

## 4. Connect Netlify

Before deploying Netlify, replace the placeholder backend URL in `netlify.toml`:

```toml
to = "https://YOUR-BACKEND-URL.onrender.com/summarize/"
```

and the health URL accordingly.

Then connect the GitHub repository to Netlify.

Build settings:

- Publish directory: `frontend`
- Build command: leave empty

After deployment, the browser calls `/api/summarize`, and Netlify proxies that request to the Render API.

## Local development

Run the backend:

```bash
cd backend
py -3.12 -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Serve the frontend with a simple static server from the repository root, for example:

```bash
py -3.12 -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500`.

## Important model note

The original uploaded application did not contain the trained model directory. The API therefore uses `MODEL_ID` so the deployed service can load the trained model from the Hugging Face Hub instead of committing model weights into the frontend repository.
