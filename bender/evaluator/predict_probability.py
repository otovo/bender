import logging

import plotly.express as px

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedClassificationModel, TrainedModel

logger = logging.getLogger(__name__)


class ProbabilityForClassification(Evaluator):

    classification_of_interest: str
    exporter: Exporter

    def __init__(self, exporter: Exporter, classification_of_interest: str) -> None:
        self.exporter = exporter
        self.classification_of_interest = classification_of_interest

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        if not isinstance(model, TrainedClassificationModel):
            logger.error('ProbabilityForClassification Evaluator will only work for TrainedClassificationModel')
            return
        y_score = model.predict_proba(data_set.x_validate)[self.classification_of_interest]
        # Scores compared to true labels
        fig_hist = px.histogram(
            x=y_score,
            color=data_set.y_validate,
            nbins=50,
            labels={'color': f'{self.classification_of_interest} Label', 'x': 'Score', 'width': 100, 'height': 200},
        )
        await self.exporter.store_figure(fig_hist)
