from typing import Any, List, Protocol

import pandas as pd
from loguru import logger
from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split


class TrainParams(BaseModel):
    data_path: str
    label: str = Field(default="target")
    train_size: float = Field(default=0.8, gt=0, lt=1)


class LinearBaseModel(Protocol):
    def __init__(self):
        self.model: Any
        self.logger: Any

    @logger.catch
    def _prepare_data(self, data_path: str, label: str, train_size: float) -> List:
        df: DataFrame = pd.read_csv(data_path)
        self.logger.info(
            f"Read data from {data_path}; It has {df.shape[0]} rows, {df.shape[1]} ftrs, and {label} as label"
        )

        return train_test_split(
            df.drop(label, axis=1), df[label], train_size=train_size
        )

    def train(self, train_params: TrainParams) -> None:
        ...

    def predict(self, data: DataFrame) -> None:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...
