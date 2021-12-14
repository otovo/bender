from pandas.core.frame import DataFrame, Series
from bender.evaluators import Evaluators
from bender.importers import DataImporters

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from bender.model_trainer import ModelTrainer, XGBoostTrainer
from bender.split_strategy import SplitStrategy
from bender.transformation import UnpackTypePolicy

from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio

@pytest.fixture
def json_df() -> DataFrame:
    return DataFrame(
        {
            'json_data': [
                # Different json formatting is intentional
                # Therefore testing if the regex is correct
                '[{"value": 1},{"value": 1},{"value":2, "other": 10},{"value": 4}]',
                '[{"value":   1},{"value": 1},{"value": 1},{"value": 1}]',
            ],
            'value_median': [1.5, 1],
            'value_mean': [2, 1],
            'value_min': [1, 1],
            'value_max': [4, 1],
        }
    ).copy()

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

async def test_training_pipeline(input_data):

    days = Series([20, 21, 22, 24, 25, 27, 28, 2, 9])
    months = Series([1, 2, 3, 4, 5, 6, 7, 8, 9])

    pipeline = (DataImporters
        .literal(input_data)
        .process([
            Transformations.log_normal_shift("y_values", "y_log", input_has_zeros=False),
            Transformations.neighour_distance(number_of_neighbours=2, latitude="lat", longitude="long"),
            Transformations.date_component("day", "date", output_feature="day_value"),
            Transformations.date_component("month", "date", output_feature="month_value"),
            Transformations.date_component("year", "date", output_feature="year_value")
        ])
    )
    output_df = await pipeline.run()
    assert_almost_equal(output_df['x_values'].to_numpy(), output_df['y_log'].to_numpy())
    assert np.all(output_df['day_value'] == days)
    assert np.all(output_df['month_value'] == months)
    assert np.all(output_df['year_value'] == Series([2020] * len(input_data)))



async def run_test_for(policy: UnpackTypePolicy, name: str, json_df: DataFrame) -> None:
    transformation = Transformations.unpack_json('json_data', key='value', output_feature=f'out_{name}', policy=policy)
    ret_df = await transformation.transform(json_df)
    assert np.all(ret_df[f'out_{name}'] == ret_df[f'value_{name}'])


async def test_unpack_json_median(json_df: DataFrame) -> None:
    await run_test_for(UnpackTypePolicy.median_number(), 'median', json_df)


async def test_unpack_json_mean(json_df: DataFrame) -> None:
    await run_test_for(UnpackTypePolicy.mean_number(), 'mean', json_df)


async def test_unpack_json_min(json_df: DataFrame) -> None:
    await run_test_for(UnpackTypePolicy.min_number(), 'min', json_df)


async def test_unpack_json_max(json_df: DataFrame) -> None:
    await run_test_for(UnpackTypePolicy.max_number(), 'max', json_df)