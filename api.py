"""Standalone API: anonymize a text with the chosen model. Run: uv run uvicorn api:app"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from anonymize import anonymize
from detectors import DETECTORS, get_detector

MAX_CHARS = 1000  # ~250 tokens: Piiranha's context is 256 and nothing is chunked yet
ModelName = Literal[tuple(DETECTORS)]

@asynccontextmanager
async def lifespan(app):
    for name in DETECTORS:  # eager load: the first request is as fast as the rest
        get_detector(name)
    yield


app = FastAPI(title="Data anonymization", lifespan=lifespan)


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    model: ModelName = "GLiNER"


class EntityOut(BaseModel):
    text: str
    label: str
    start: int
    end: int


class Response(BaseModel):
    anonymized_text: str
    entities: list[EntityOut]


@app.get("/models")
def models() -> list[str]:
    return list(DETECTORS)


# plain def: FastAPI runs it in a threadpool, inference is CPU-bound
@app.post("/anonymize")
def post_anonymize(req: Request) -> Response:
    entities = get_detector(req.model).detect(req.text)
    return Response(anonymized_text=anonymize(req.text, entities), entities=[EntityOut(**vars(e)) for e in entities])
