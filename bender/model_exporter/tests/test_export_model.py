import pytest
from pandas import DataFrame

from bender.importers import DataImporters
from bender.model_exporters import ModelExporters
from bender.model_loaders import ModelLoaders
from bender.model_trainer.xgboosted_tree import XGBoostTrainer
from bender.split_strategies import SplitStrategies

pytestmark = pytest.mark.asyncio


async def test_export_model_pipeline(date_df: DataFrame) -> None:

    model, data_set = await (
        DataImporters.literal(date_df)
        .split(SplitStrategies.ratio(1))
        .train(XGBoostTrainer(), input_features=['y_values', 'x_values'], target_feature='classification')
        .export_model(ModelExporters.disk('test-exports/test_file.json'))
        .run()
    )

    _, _, result = await (DataImporters.literal(date_df).load_model(ModelLoaders.literal(model)).predict().run())

    assert len(result) == len(date_df)
