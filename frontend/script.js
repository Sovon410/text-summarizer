const form = document.getElementById("summarization-form");
const textInput = document.getElementById("text-input");
const summaryOutput = document.getElementById("summary-output");
const summaryText = document.getElementById("summary-text");
const submitButton = document.getElementById("submit-button");
const submitLabel = document.getElementById("submit-label");
const loadingSpinner = document.getElementById("loading-spinner");
const clearButton = document.getElementById("clear-button");
const errorMessage = document.getElementById("error-message");
const characterCount = document.getElementById("character-count");

const LOCAL_API_URL = "http://127.0.0.1:8000/summarize/";

const PRODUCTION_API_URL =
    "https://text-summarizer-api-gq03.onrender.com/summarize/";

const API_URL =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
        ? LOCAL_API_URL
        : PRODUCTION_API_URL;

function setLoading(isLoading) {
    submitButton.disabled = isLoading;
    clearButton.disabled = isLoading;
    loadingSpinner.hidden = !isLoading;
    submitLabel.textContent = isLoading ? "Summarizing..." : "Summarize";
    summaryOutput.setAttribute("aria-busy", String(isLoading));
}

function showSummary(summary) {
    summaryText.textContent = summary;
    summaryOutput.hidden = false;
}

function clearSummary() {
    summaryText.textContent = "";
    summaryOutput.hidden = true;
    summaryOutput.setAttribute("aria-busy", "false");
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.hidden = true;
}

function updateCharacterCount() {
    characterCount.textContent =
        `${textInput.value.length.toLocaleString()} / 20,000`;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const dialogue = textInput.value.trim();

    if (!dialogue) {
        clearSummary();
        showError("Please enter some text to summarize.");
        textInput.focus();
        return;
    }

    clearError();
    clearSummary();
    setLoading(true);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ dialogue })
        });

        let data = null;

        try {
            data = await response.json();
        } catch {
            data = null;
        }

        if (!response.ok) {
            const detail =
                data?.detail || `Server error (${response.status}).`;

            throw new Error(detail);
        }

        const summary = data?.summary?.trim();

        if (!summary) {
            throw new Error("The server returned an empty summary.");
        }

        showSummary(summary);

    } catch (error) {
        console.error("Summarization request failed:", error);

        showError(
            error instanceof Error
                ? error.message
                : "Unable to summarize the text."
        );
    } finally {
        setLoading(false);
    }
});

clearButton.addEventListener("click", () => {
    textInput.value = "";
    clearSummary();
    clearError();
    updateCharacterCount();
    textInput.focus();
});

textInput.addEventListener("input", () => {
    clearError();
    updateCharacterCount();
});

updateCharacterCount();

