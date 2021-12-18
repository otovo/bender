import pytest

from bender.evaluators import Evaluators
from bender.explorers import Explorers
from bender.exporter.exporter import LocalDiskExporter
from bender.importers import DataImporters, DataSets
from bender.metrics import Metrics
from bender.model_trainers import Trainers
from bender.split_strategies import SplitStrategies
from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio


async def test_histogram_explorer() -> None:

    input_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    _ = await (
        DataImporters.data_set(DataSets.IRIS)
        .explore([Explorers.histogram(input_features, exporter=LocalDiskExporter('test-exports/explorer'))])
        .run()
    )


async def test_histogram_with_target_explorer() -> None:

    input_features = ['pl exp', 'pw exp', 'petal length (cm)']

    exporter = LocalDiskExporter('test-exports/explorer')

    _ = await (
        DataImporters.data_set(DataSets.IRIS)
        .process(
            [
                Transformations.exp_shift('petal length (cm)', output='pl exp'),
                Transformations.exp_shift('petal width (cm)', output='pw exp'),
            ]
        )
        .explore(
            [
                Explorers.histogram(input_features, target='target', exporter=exporter),
                Explorers.correlation(input_features, exporter),
                Explorers.pair_plot('target', input_features, exporter),
                Explorers.scatter('pl exp', 'pw exp', target='target', exporter=exporter),
            ]
        )
        .split(SplitStrategies.uniform_ratio('target', 0.7))
        .train(Trainers.kneighbours(), input_features=input_features, target_feature='target')
        .evaluate(
            [
                Evaluators.confusion_matrix(exporter),
                Evaluators.precision_recall(exporter),
                Evaluators.roc_curve(exporter),
            ]
        )
        .metric(Metrics.log_loss())
        .run()
    )
