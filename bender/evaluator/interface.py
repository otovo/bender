from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel


class Evaluator:
    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        """Evaluates a model

        Args:
            model ([type]): The model to evaluate
            data_set (DataSplit): The data that can be used to evaluate the model
        """
        raise NotImplementedError()
