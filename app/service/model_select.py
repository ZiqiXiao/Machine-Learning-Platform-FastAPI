from config.settings import MissionTypes, ModelTypes
from ml.linear import LinearBM, LinearRegression


def model_select(mission: MissionTypes, model: ModelTypes) -> LinearBM.LinearBaseModel:
    if mission == MissionTypes.LINEAR_REGRESSION:
        if model == ModelTypes.LR_LINEAR_REGRESSION:
            return LinearRegression.LinearRegressionModel()
