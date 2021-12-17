import pytest
from sklearn.datasets import load_iris

from bender.importers import DataImporters
from bender.metrics import Metrics
from bender.model_exporters import ModelExporters
from bender.model_trainer.xgboosted_tree import XGBoostTrainer
from bender.split_strategies import SplitStrategies

pytestmark = pytest.mark.asyncio


async def test_log_loss_metric() -> None:

    iris = load_iris(as_frame=True)

    data = iris.data
    data['target'] = iris.target

    loss = await (
        DataImporters.literal(data)
        .split(SplitStrategies.ratio(0.7))
        .train(
            XGBoostTrainer(),
            input_features=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
            target_feature='target',
        )
        .export_model(ModelExporters.disk('test-exports/test_file.json'))
        .metric(Metrics.log_loss())
        .run()
    )

    assert isinstance(loss, float)
    assert loss > 0
