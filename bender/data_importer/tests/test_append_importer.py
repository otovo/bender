import numpy as np
import pytest
from pandas.core.frame import DataFrame

from bender.importers import DataImporters

pytestmark = pytest.mark.asyncio


async def test_append_importer(date_df: DataFrame) -> None:

    index = 3
    first = date_df[:index]
    second = date_df[index:]
    df = await DataImporters.literal(first).append(DataImporters.literal(second)).run()

    assert np.all(df == date_df)
