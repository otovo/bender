from sklearn.metrics import mean_absolute_percentage_error

from bender.metric.interface import Metric
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedClassificationModel, TrainedModel


class MeanAbsolutePercentageError(Metric):
    async def metric(self, model: TrainedModel, data_set: TrainingDataSet) -> float:
        if isinstance(model, TrainedClassificationModel):
            raise Exception('Mean Square error is only for Regression models')
        else:
            result = model.predict(data_set.x_validate)
            return float(mean_absolute_percentage_error(data_set.y_validate, result))
