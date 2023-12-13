import os

import pytest

from app.models import ml
from ml.linear import LinearBM
from ml.linear.LinearRegression import LinearRegressionModel


@pytest.fixture
def model() -> LinearRegressionModel:
    model = LinearRegressionModel()
    model._uid = "1234_1234_1234"
    return model


class TestLinearRegression:
    def test_prepare_data(self, model: LinearRegressionModel) -> None:
        X_train, X_test, y_train, y_test = model._prepare_data(
            data_path="test/ml/data/data.csv", label="target", train_size=0.8
        )
        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert y_train.shape[0] == 8
        assert y_test.shape[0] == 2

    def test_train(self, model: LinearRegressionModel) -> None:
        train_params = LinearBM.TrainParams(
            data_path="ml/linear/diabetes.csv", label="target", train_size=0.8
        )
        model.train(train_params)

    def test_save(self, model: LinearRegressionModel) -> ml.ModelInfo:
        name: str = "test_model"
        save_path: str = os.path.join(
            model.PROJECT_ROOT, "test", "ml", "data", str(model._uid)
        )

        model_save = model.save(name=name, save_path=save_path)

        assert os.path.exists(model_save.path)
        assert model_save.name == name
        assert model_save.uid == str(model._uid)
        assert model_save.path == save_path

        return model_save
