import os

from ml.linear import LinearBM
from ml.linear.LinearRegression import LinearRegressionModel


def test_prepare_data() -> None:
    model = LinearRegressionModel()
    X_train, X_test, y_train, y_test = model._prepare_data(
        data_path="test/ml/data/data.csv", label="target", train_size=0.8
    )
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2


# TODO: Need find a way to use assertion.
def test_train() -> None:
    model = LinearRegressionModel()
    train_params = LinearBM.TrainParams(
        data_path="ml/linear/diabetes.csv", label="target", train_size=0.8
    )
    model.train(train_params)
