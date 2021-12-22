import numpy as np
from sklearn.metrics import log_loss

from bender.metric.interface import Metric
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel, TrainedProbabilisticClassificationModel


class LogLoss(Metric):
    async def metric(self, model: TrainedModel, data_set: TrainingDataSet) -> float:
        if isinstance(model, TrainedProbabilisticClassificationModel):
            y_pred = model.predict_proba(data_set.x_validate)
            return float(log_loss(data_set.y_validate, y_pred, labels=model.class_names()))
        else:
            result = model.predict(data_set.x_validate)
            return float(np.log1p(abs(result - data_set.y_validate)).mean())
