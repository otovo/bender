import pytest
from pandas.core.frame import DataFrame

from bender.evaluators import Evaluators
from bender.exporter.exporter import Exporter
from bender.importers import DataImporters
from bender.model_trainer.xgboosted_tree import XGBoostTrainer
from bender.split_strategies import SplitStrategies
from bender.trained_model.xgboosted_tree import TrainedXGBoostModel
from bender.transformation.transformation import BinaryTransform
from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio


async def test_train_pipeline(date_df: DataFrame) -> None:

    pipeline = (
        DataImporters.literal(date_df)
        .process(
            [
                Transformations.log_normal_shift('y_values', 'y_log', input_has_zeros=False),
                Transformations.neighour_distance(number_of_neighbours=2, latitude='lat', longitude='long'),
                Transformations.date_component('day', 'date', output_feature='day_value'),
                Transformations.date_component('month', 'date', output_feature='month_value'),
                Transformations.date_component('year', 'date', output_feature='year_value'),
                BinaryTransform('target', lambda df: df['y_values'] > 2),
            ]
        )
        .split(SplitStrategies.ratio(0.7))
        .train(XGBoostTrainer(), input_features=['x_values', 'day_value', 'month_value'], target_feature='target')
        .evaluate(
            [
                Evaluators.roc_curve(Exporter.disk('test-exports/train-evaluation-roc.png')),
                Evaluators.confusion_matrix(Exporter.disk('test-exports/train-evaluation-matrix.png')),
                Evaluators.precision_recall(Exporter.disk('test-exports/train-evaluation-prec-recall.png')),
            ]
        )
    )
    model, _ = await pipeline.run()
    assert isinstance(model, TrainedXGBoostModel)
