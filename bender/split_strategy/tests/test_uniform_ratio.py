import pytest
from pandas import DataFrame

from bender.importers import DataImporters
from bender.split_strategies import SplitStrategies

pytestmark = pytest.mark.asyncio


async def test_start_ratio(date_df: DataFrame) -> None:

    ratio = 0.8

    for offset in range(0, 10):
        train, test = await (
            DataImporters.literal(date_df)
            .split(SplitStrategies.uniform_ratio('bool_classification', ratio, start_offset=offset / 10))
            .run()
        )
