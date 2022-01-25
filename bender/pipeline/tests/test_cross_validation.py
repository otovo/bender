import pytest
from pandas import DataFrame

from bender.importers import DataImporters
from bender.model_trainer.xgboosted_tree import XGBoostTrainer

pytestmark = pytest.mark.asyncio


async def test_cross_validation(date_df: DataFrame) -> None:

    _ = await (
        DataImporters.literal(date_df)
        .cross_validate(
            'classification',
            3,
            lambda pipeline: pipeline.train(
                XGBoostTrainer(), input_features=['y_values', 'x_values'], target_feature='bool_classification'
            ),
        )
        .run()
    )
