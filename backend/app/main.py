import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .summarizer import summarize_text


app = FastAPI(
    title="Text Summarizer API",
    description="Text summarization API powered by a fine-tuned T5 model.",
    version="2.0.0",
)


# --------------------------------------------------
# CORS Configuration
# --------------------------------------------------

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,"
        "http://localhost:5500,"
        "https://text-summarizer-t5-small.netlify.app",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class DialogueInput(BaseModel):
    dialogue: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
    )

    @field_validator("dialogue")
    @classmethod
    def validate_dialogue(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Dialogue cannot be empty.")

        return value


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }


# --------------------------------------------------
# Summarization Endpoint
# --------------------------------------------------

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput) -> dict[str, str]:

    try:
        summary = summarize_text(dialogue_input.dialogue)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Do not expose model internals to the browser.
        raise HTTPException(
            status_code=500,
            detail="The summarization service failed while generating the summary.",
        ) from exc

    return {
        "summary": summary
    }