from cmath import isnan

import pytest
from pandas import DataFrame

from bender.importers import DataImporters
from bender.metrics import Metrics
from bender.model_trainer.xgboosted_tree import XGBoostTrainer

pytestmark = pytest.mark.asyncio


async def test_cross_validation(date_df: DataFrame) -> None:

    score = await (
        DataImporters.literal(date_df)
        .cross_validate(
            'classification',
            3,
            lambda pipeline: pipeline.train(
                XGBoostTrainer(), input_features=['y_values', 'x_values'], target_feature='bool_classification'
            ).metric(Metrics.log_loss()),
        )
        .run()
    )

    assert score > 0


async def test_cross_validation_nan_metric(date_df: DataFrame) -> None:

    score = await (
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

    assert isnan(score)
