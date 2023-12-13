import os
from logging import Logger
from uuid import UUID

import joblib
import loguru
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.models import ml
from config.settings import ml_logger as logger
from ml.linear import LinearBM


# TODO: Add WebSocket support
# TODO: Unique ID for each request
class LinearRegressionModel(LinearBM.LinearBaseModel):
    def __init__(self):
        super().__init__()
        self.model: LinearRegression = LinearRegression()
        self.logger: Logger = logger

    # TODO: Metrics
    @loguru.logger.catch(reraise=True)
    def train(self, train_params: LinearBM.TrainParams):
        self.logger.info("Start training")
        X_train, X_test, y_train, y_test = self._prepare_data(
            train_params.data_path,
            train_params.label,
            train_params.train_size,
        )
        self.model.fit(X_train, y_train)
        self.logger.success("Training finished")

    @loguru.logger.catch(reraise=True)
    def predict(self, pred_params: LinearBM.PredictParams) -> np.ndarray:
        self.logger.info("Start predicting")
        self.model: LinearRegression = joblib.load(pred_params.path_to_model)
        self.logger.info(f"Model loaded from {pred_params.path_to_model}")
        df: pd.DataFrame = pd.read_csv(pred_params.data_path)
        predicted = self.model.predict(df)
        self.logger.success("Predicting finished")
        return predicted

    @loguru.logger.catch(reraise=True)
    def save(
        self,
        name: str | None = None,
        save_path: str | None = None,
    ) -> ml.ModelInfo:
        if not save_path:
            save_path = os.path.join(self.PROJECT_ROOT, "data", "ml", str(self._uid))
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(
            self.model, os.path.join(save_path, f"{name if name else 'model'}.joblib")
        )
        self.logger.info(f"Model saved {save_path}")

        model_save: ml.ModelInfo = ml.ModelInfo(
            uid=str(self._uid),
            path=save_path,
            name=name,
        )
        return model_save
