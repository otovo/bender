from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.core.frame import DataFrame, Series

from bender.importers import DataImporters
from bender.transformation.schema import SchemaType
from bender.transformation.transformation import UnpackPolicy
from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio


async def test_training_pipeline(date_df: DataFrame) -> None:

    pipeline = DataImporters.literal(date_df).process(
        [
            Transformations.log_normal_shift('y_values', 'y_log', input_has_zeros=False),
            Transformations.neighour_distance(number_of_neighbours=2, latitude='lat', longitude='long'),
            Transformations.date_component('day', 'date', output='day_value'),
            Transformations.date_component('month', 'date', output='month_value'),
            Transformations.date_component('year', 'date', output='year_value'),
        ]
    )
    output_df = await pipeline.run()
    assert_almost_equal(output_df['x_values'].to_numpy(), output_df['y_log'].to_numpy())
    assert np.all(output_df['day_value'] == output_df['expected_day'])
    assert np.all(output_df['month_value'] == output_df['expected_month'])
    assert np.all(output_df['year_value'] == Series([2020] * len(date_df)))


async def run_test_for(policy: UnpackPolicy, name: str, json_df: DataFrame) -> None:
    transformation = Transformations.unpack_json('json_data', key='value', output=f'out_{name}', policy=policy)
    ret_df = await transformation.transform(json_df)
    assert np.all(ret_df[f'out_{name}'] == ret_df[f'value_{name}'])


async def test_unpack_json_median(json_df: DataFrame) -> None:
    await run_test_for(UnpackPolicy.median_number(), 'median', json_df)


async def test_unpack_json_mean(json_df: DataFrame) -> None:
    await run_test_for(UnpackPolicy.mean_number(), 'mean', json_df)


async def test_unpack_json_min(json_df: DataFrame) -> None:
    await run_test_for(UnpackPolicy.min_number(), 'min', json_df)


async def test_unpack_json_max(json_df: DataFrame) -> None:
    await run_test_for(UnpackPolicy.max_number(), 'max', json_df)


async def test_date_component_handle_object_type() -> None:
    input_data = pd.read_csv('test-data/dates.csv')
    result = await Transformations.date_component('day', 'date', 'day_value').transform(input_data)
    assert np.all(result['expected_day'] == result['day_value'])


async def test_date_component_handle_str_type(date_df: DataFrame) -> None:
    date_df['date'] = date_df['date'].astype(str)
    result = await Transformations.date_component('day', 'date', 'day_value').transform(date_df)
    assert np.all(result['expected_day'] == result['day_value'])


async def test_date_component_handle_datetime_type(date_df: DataFrame) -> None:
    date_df['date'] = date_df['date'].apply(lambda date: datetime.fromisoformat(date))
    result = await Transformations.date_component('day', 'date', 'day_value').transform(date_df)
    assert np.all(result['expected_day'] == result['day_value'])


async def test_schema(date_df: DataFrame) -> None:
    schema = {'date': SchemaType.datetime(), 'x_values': SchemaType.integer()}
    result_df = await Transformations.schema(schema).transform(date_df)

    for (key, value) in schema.items():
        assert result_df[key].dtype == value.data_type
