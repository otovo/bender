import numpy as np
import pytest
from pandas.core.frame import DataFrame

from bender.evaluators import Evaluators
from bender.exporter.exporter import Exporter
from bender.importers import DataImporters
from bender.split_strategies import SplitStrategies
from bender.trainer.model_trainer import TrainedXGBoostModel, XGBoostTrainer
from bender.transformation.transformation import BinaryTransform
from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio


@pytest.fixture  # type: ignore
def input_data() -> DataFrame:
    values = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4], dtype=float)
    return DataFrame(
        {
            'x_values': values,
            'y_values': np.exp(values),
            'date': [
                '20-01-2020',
                '21-02-2020',
                '22-03-2020',
                '24-04-2020',
                '25-05-2020',
                '27-06-2020',
                '28-07-2020',
                '02-08-2020',
                '09-09-2020',
            ],
            'lat': values,
            'long': values,
        }
    )


async def test_train_pipeline(input_data: DataFrame) -> None:

    pipeline = (
        DataImporters.literal(input_data)
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
    model = await pipeline.run()
    assert isinstance(model, TrainedXGBoostModel)
