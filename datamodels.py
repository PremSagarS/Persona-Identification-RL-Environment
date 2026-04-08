from pydantic import Field, BaseModel, field_validator
from typing import Literal, List

import json

class ProductReview(BaseModel):
    title: str
    rating: float
    price: float | None
    description: str
    review_text: str

class Product(BaseModel):
    title: str
    price: float | None
    description: str

class PersonaPrediction(BaseModel):
    persona: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def must_be_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be [0, 1], got {v}")
        return v

class Persona(BaseModel):
    name: str
    description: str
    signals: List[str]