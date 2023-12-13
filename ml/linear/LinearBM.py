import os
import uuid
from logging import Logger
from typing import Any, List, Protocol
from uuid import UUID

import loguru
import pandas as pd
from icecream import ic
from pandas import DataFrame
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from app.models import ml
from config.settings import PROJECT_ROOT, MissionTypes, ModelTypes


class BasicParams(BaseModel):
    mission: MissionTypes
    model: ModelTypes


class TrainParams(BaseModel):
    data_path: str
    label: str = Field(default="target")
    train_size: float = Field(default=0.8, gt=0, lt=1)


class PredictParams(BaseModel):
    data_path: str
    path_to_model: str


class LinearBaseModel(Protocol):
    """All Linear Regression Model should implement this protocol."""

    model: Any
    _uid: UUID
    PROJECT_ROOT: str = PROJECT_ROOT
    _logger_id: int
    logger: Logger
    _train_store_path: str

    # TODO - instance should log to their own folder.
    def __init__(self) -> None:
        self._uid = uuid.uuid4()
        self._train_store_path = os.path.join(
            self.PROJECT_ROOT, "data", "ml", str(self._uid)
        )
        # self._logger_id = loguru.logger.add(
        #     os.path.join(self._train_store_path, "model.log"),
        #     format="{time:YYYY-MM-DD HH:mm:ss.SSS} {extra[uid]} | {level: <8} | {name}:{function}:{line} - {message}",
        #     level="DEBUG",
        #     filter=lambda record: record["extra"].get("uid") == str(self._uid),
        # )
        # self.logger = loguru.logger.bind(type="ml", uid=str(self._uid))

    @loguru.logger.catch
    def _prepare_data(self, data_path: str, label: str, train_size: float) -> List:
        df: DataFrame = pd.read_csv(data_path)
        self.logger.info(
            f"Read data from {data_path}; It has {df.shape[0]} rows, {df.shape[1]} ftrs, and {label} as label"
        )

        return train_test_split(
            df.drop(label, axis=1), df[label], train_size=train_size
        )

    # @loguru.logger.catch
    # def __del__(self):
    #     log_dir: str = os.path.join(self.PROJECT_ROOT, "data", "ml", str(self._uid))
    #     if os.path.exists(log_dir) and all(
    #         fname.endswith(".log") for fname in os.listdir(log_dir)
    #     ):
    #         shutil.rmtree(log_dir)
    #     if self._logger_id in self.logger._core.handlers:
    #         self.logger.remove(self._logger_id)

    def train(self, train_params: TrainParams) -> None:
        ...

    def predict(self, pred_params: PredictParams) -> DataFrame:
        ...

    def save(self) -> ml.ModelInfo:
        ...
