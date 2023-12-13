from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    uid: str = Field(min_length=1)
    path: str = Field(min_length=1)
    name: str = Field(min_length=1, default=str(uid))
