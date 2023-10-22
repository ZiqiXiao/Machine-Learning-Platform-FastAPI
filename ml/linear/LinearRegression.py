from sklearn.linear_model import LinearRegression

from config.settings import ml_logger as logger
from ml.linear import LinearBM


class LinearRegressionModel(LinearBM.LinearBaseModel):
    def __init__(self):
        self.model = LinearRegression()
        self.logger = logger

    @logger.catch(reraise=True)
    def train(self, train_params: LinearBM.TrainParams):
        self.logger.info("Start training")
        X_train, X_test, y_train, y_test = self._prepare_data(
            train_params.data_path,
            train_params.label,
            train_params.train_size,
        )
        self.model.fit(X_train, y_train)
        self.logger.success("Training finished")

    def predict(self, data):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
