from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


class ChatSource(BaseModel):
    document: str
    page: int


class ChatResponse(BaseModel):
    reponse: str
    sources: list[ChatSource] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    document: str
    pages: int
    chunks: int
    indexed_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DocumentIndexResponse(BaseModel):
    document: str
    pages: int
    chunks: int
