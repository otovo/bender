from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel


class Metric:
    async def metric(self, model: TrainedModel, data_set: TrainingDataSet) -> float:
        raise NotImplementedError()
