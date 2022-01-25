import os
from logging.config import dictConfig

import numpy as np
import pytest
from pandas import DataFrame


def log_level(non_prod_value: str, prod_value: str) -> str:
    """
    Helper function for setting an appropriate log level in prod.
    """
    return prod_value


@pytest.fixture  # type: ignore
def date_df() -> DataFrame:
    values = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4, 4], dtype=float)
    return DataFrame(
        {
            'x_values': values,
            'y_values': np.exp(values),
            'date': [
                '2020-01-20T01:09:07',
                '2020-02-21T01:09:07',
                '2020-03-22T01:09:07',
                '2020-04-24T01:09:07',
                '2020-05-25T01:09:07',
                '2020-06-27T01:09:07',
                '2020-07-28T01:09:07',
                '2020-08-02T01:09:07',
                '2020-09-09T01:09:07',
                '2020-10-09T01:09:07',
            ],
            'lat': values,
            'long': values,
            'expected_day': [20, 21, 22, 24, 25, 27, 28, 2, 9, 3],
            'expected_month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
            'classification': [0, 0, 1, 3, 2, 2, 1, 3, 3, 3],
            'bool_classification': [True, True, True, False, False, True, False, False, True, False],
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

handler = 'console'


def configure_logging() -> None:
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'console': {
                    'class': 'logging.Formatter',
                    'datefmt': '%H:%M:%S',
                    'format': '%(levelname)s:\t\b%(asctime)s %(name)s:%(lineno)d %(message)s',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'console',
                },
            },
            'loggers': {
                # project
                'bender': {'handlers': [handler], 'level': log_level('INFO', 'DEBUG'), 'propagate': True},
                # third-party packages
                'arq': {'handlers': [handler], 'level': log_level('WARNING', 'INFO'), 'propagate': True},
                'faker': {'handlers': [handler], 'level': 'INFO'},
                'httpx': {'handlers': [handler], 'level': 'INFO'},
                'oauthlib': {'handlers': [handler], 'level': 'INFO'},
            },
        }
    )


configure_logging()
