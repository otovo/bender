from sklearn.linear_model import LinearRegression

from bender.model_trainer.interface import ModelTrainer
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel
from bender.trained_model.linear_regression import TrainedLinearRegression


class LinearRegressionTrainer(ModelTrainer):
    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        lin_reg = LinearRegression()
        lin_reg.fit(data_split.x_train, data_split.y_train)
        return TrainedLinearRegression(lin_reg, data_split.x_features)
