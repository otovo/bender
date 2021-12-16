import pytest
from pandas.core.frame import DataFrame

from bender.importers import DataImporters
from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio


async def test_cached_importer(date_df: DataFrame) -> None:
    cache_path = 'test-cache'
    processing = [Transformations.log_normal_shift('y_values', 'y_log_values')]
    original_data = await DataImporters.literal(date_df).cached(cache_path).process(processing).run()
    cached_data = await DataImporters.literal(DataFrame({})).cached(cache_path).process(processing).run()

    cached_data.set_index(original_data.index)
    assert len(cached_data.columns) == len(original_data.columns)
