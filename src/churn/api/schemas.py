from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    values: list[float] = Field(min_length=30, max_length=30)


class PredictResponse(BaseModel):
    prediction: int
