import logging

from sklearn.metrics import RocCurveDisplay

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedClassificationModel, TrainedEstimatorModel, TrainedModel

logger = logging.getLogger(__name__)


class RocCurve(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        if not isinstance(model, TrainedEstimatorModel):
            return
        if isinstance(model, TrainedClassificationModel) and len(model.class_names()) > 2:
            logger.info('RocCurve can not handle multi class classification models')
            return
        display = RocCurveDisplay.from_estimator(
            model.estimator(), data_set.x_validate, data_set.y_validate.astype(float)
        )
        _ = display.ax_.set_title('Roc Curve')
        await self.exporter.store_figure(display.figure_)
