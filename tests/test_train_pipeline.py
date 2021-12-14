from pandas.core.frame import DataFrame, Series
from bender.evaluators import Evaluators
from bender.exporter import Exporter
from bender.importers import DataImporters

import pytest
import numpy as np
from bender.model_trainer import TrainedXGBoostModel, XGBoostTrainer
from bender.split_strategy import SplitStrategy
from bender.transformation import BinaryTransform

from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio

@pytest.fixture
def input_data() -> DataFrame:
    values = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4], dtype=float)
    return DataFrame({
        'x_values' : values,
        'y_values' : np.exp(values),
        'date' : ['20-01-2020', '21-02-2020', '22-03-2020', '24-04-2020', '25-05-2020', '27-06-2020', '28-07-2020', '02-08-2020', '09-09-2020'],
        'lat' : values,
        'long' : values
    })

async def test_data_processing_pipeline(input_data):

    days = Series([20, 21, 22, 24, 25, 27, 28, 2, 9])
    months = Series([1, 2, 3, 4, 5, 6, 7, 8, 9])

    pipeline = (DataImporters
        .literal(input_data)
        .process([
            Transformations.log_normal_shift("y_values", "y_log", input_has_zeros=False),
            Transformations.neighour_distance(number_of_neighbours=2, latitude="lat", longitude="long"),
            Transformations.date_component("day", "date", output_feature="day_value"),
            Transformations.date_component("month", "date", output_feature="month_value"),
            Transformations.date_component("year", "date", output_feature="year_value"),
            BinaryTransform("target", lambda df: df['y_values'] > 2)
        ])
        .split(SplitStrategy.ratio(0.7))
        .train(XGBoostTrainer(), input_features=['x_values', 'day_value', 'month_value'], target_feature='target')
        .evaluate([
            Evaluators.roc_curve(Exporter.disk("tests/train-evaluation-roc.png")),
            Evaluators.confusion_matrix(Exporter.disk("tests/train-evaluation-matrix.png")),
            Evaluators.precision_recall(Exporter.disk("tests/train-evaluation-prec-recall.png")),
        ])
    )
    model = await pipeline.run()
    assert isinstance(model, TrainedXGBoostModel)
