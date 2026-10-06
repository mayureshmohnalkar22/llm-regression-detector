from fastapi import FastAPI, HTTPException

from app.classifier import OllamaClassifier
from app.schemas import ClassificationRequest, ClassificationResult

PROJECT_TAG = "Model Evaluation & Regression API"

app = FastAPI(
    title="LLM Regression Detector",
    description="Local-first evaluation and regression detection for an LLM support classifier.",
    version="0.2.0",
    openapi_tags=[{"name": PROJECT_TAG, "description": "Classifier and reliability checks."}],
)
classifier = OllamaClassifier()


@app.get("/health", tags=[PROJECT_TAG])
def health() -> dict[str, str]:
    return {"status": "ok", "provider": "ollama"}


@app.post("/classify", response_model=ClassificationResult, tags=[PROJECT_TAG])
def classify(request: ClassificationRequest) -> ClassificationResult:
    try:
        result, _ = classifier.classify(request.message, request.prompt_version)
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
