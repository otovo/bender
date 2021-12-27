import json

import numpy as np
import pytest
from pandas import DataFrame

from bender.importers import DataImporters
from bender.model_exporters import ModelExporters
from bender.model_loaders import ModelLoaders
from bender.model_trainer.xgboosted_tree import XGBoostTrainer
from bender.split_strategies import SplitStrategies
from bender.trained_model.xgboosted_tree import TrainedXGBoostModel

pytestmark = pytest.mark.asyncio


async def test_xgboosted_tree(date_df: DataFrame) -> None:

    model, _ = await (
        DataImporters.literal(date_df)
        .split(SplitStrategies.ratio(1))
        .train(XGBoostTrainer(), input_features=['y_values', 'x_values'], target_feature='classification')
        .export_model(ModelExporters.disk('test-exports/test_file.json'))
        .run()
    )

    json_data = model.to_json()
    json_dict = json.loads(json_data)
    loaded_model = TrainedXGBoostModel.from_dict(json_dict)

    _, _, org_result = await DataImporters.literal(date_df).load_model(ModelLoaders.literal(model)).predict().run()

    _, _, loaded_result = await (
        DataImporters.literal(date_df).load_model(ModelLoaders.literal(loaded_model)).predict().run()
    )

    assert np.all(loaded_result == org_result)
