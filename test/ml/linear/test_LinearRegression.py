import os

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
