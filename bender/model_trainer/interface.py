from typing import Generic, TypeVar

from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel


class ModelTrainer:
    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        raise NotImplementedError()


TrainableType = TypeVar('TrainableType')


class Trainable(Generic[TrainableType]):
    def train(self, model: ModelTrainer, input_features: set[str], target_feature: str) -> TrainableType:
        raise NotImplementedError()
