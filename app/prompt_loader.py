from functools import lru_cache
from pathlib import Path
import yaml

from app.config import PROMPTS_DIR


@lru_cache(maxsize=16)
def load_prompt(version: str) -> dict:
    path: Path = PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown prompt version: {version}")
    with path.open(encoding="utf-8") as file:
        prompt = yaml.safe_load(file)
    if not isinstance(prompt, dict) or "system" not in prompt:
        raise ValueError(f"Prompt file {path.name} must contain a system field")
    return prompt
