import os

import numpy as np
import pytest
from pandas import DataFrame


@pytest.fixture  # type: ignore
def date_df() -> DataFrame:
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
            'expected_day': [20, 21, 22, 24, 25, 27, 28, 2, 9],
            'expected_month': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    ).copy()


@pytest.fixture  # type: ignore
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


test_exports_path = 'test-exports'

if not os.path.isdir(test_exports_path):
    os.mkdir(test_exports_path)
