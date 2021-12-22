import pytest

from bender.evaluator.difference_distribution import AbsolutePercentageError, DifferenceDistribution
from bender.explorer.histogram import HistogramConfig, HistogramMultiple
from bender.explorers import Explorers
from bender.exporter.exporter import Exporter
from bender.importers import DataImporters, DataSets
from bender.metrics import Metrics
from bender.model_trainers import Trainers
from bender.split_strategies import SplitStrategies
from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio


async def test_train_regression_model() -> None:

    loss = await (
        DataImporters.data_set(DataSets.CALIFORNIA_HOUSING_PRICES)
        .process([Transformations.bin('target', 4, 'target_bin')])
        .explore(
            [
                Explorers.histogram(
                    target='target_bin',
                    config=HistogramConfig(n_bins=25, multiple=HistogramMultiple.FILL),
                    exporter=Exporter.disk('test-exports/hist-explor'),
                )
            ]
        )
        .split(SplitStrategies.uniform_ratio('target_bin', 0.7))
        .train(Trainers.linear_regression(), input_features=['MedInc', 'Population'], target_feature='target')
        .evaluate([DifferenceDistribution(AbsolutePercentageError(), Exporter.disk('test-exports/diff-dist'))])
        .metric(Metrics.mean_absolute_percentage_error())
        .run()
    )
    assert loss > 0
