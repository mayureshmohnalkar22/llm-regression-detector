import json
import time
import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.prompt_loader import load_prompt
from app.schemas import ClassificationResult


class OllamaClassifier:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def classify(self, message: str, prompt_version: str = "v1") -> tuple[ClassificationResult, float]:
        prompt = load_prompt(prompt_version)
        payload = {
            "model": self.model,
            "stream": False,
            "format": ClassificationResult.model_json_schema(),
            "options": {"temperature": prompt.get("temperature", 0)},
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": f"Classify this support message:\\n{message}"},
            ],
        }
        start = time.perf_counter()
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=90)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("Could not reach Ollama. Start Ollama and pull the configured model.") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        try:
            content = response.json()["message"]["content"]
            result = ClassificationResult.model_validate(json.loads(content))
        except (KeyError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("Ollama returned invalid structured output") from exc
        return result, latency_ms
