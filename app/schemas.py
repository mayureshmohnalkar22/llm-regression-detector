from enum import Enum
from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassificationRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    prompt_version: str = "v1"


class ClassificationResult(BaseModel):
    category: Category
    summary: str = Field(min_length=1, max_length=280)
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1, max_length=500)
