import plotly.express as px

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trainer.model_trainer import TrainedModel


class PredictProbability(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        y_score = model.predict_proba(data_set.x_validate)[:, 1]
        # Scores compared to true labels
        fig_hist = px.histogram(
            x=y_score,
            color=data_set.y_validate,
            nbins=50,
            labels={'color': 'True Label', 'x': 'Score', 'width': 100, 'height': 200},
        )
        await self.exporter.store_figure(fig_hist)

        # self.logger.report_plotly(title="p(x|y = 'lost' or 'won')",series='status',figure=fig_hist)
