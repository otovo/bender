from sklearn.metrics import ConfusionMatrixDisplay

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedEstimatorModel, TrainedModel


class ConfusionMatrix(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        if not isinstance(model, TrainedEstimatorModel):
            return
        display = ConfusionMatrixDisplay.from_estimator(
            model.estimator(), data_set.x_validate, data_set.y_validate.astype(float)
        )
        _ = display.ax_.set_title('Confusion Matrix')
        await self.exporter.store_figure(display.figure_)
