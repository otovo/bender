import pytest
from sklearn.datasets import load_iris

from bender.explorers import Explorers
from bender.importers import DataImporters
from bender.transformations import Transformations

pytestmark = pytest.mark.asyncio


async def test_histogram_explorer() -> None:

    iris = load_iris(as_frame=True)

    data = iris.data
    data['target'] = iris.target

    input_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    _ = await (DataImporters.literal(data).explore([Explorers.histogram(input_features)]).run())


async def test_histogram_with_target_explorer() -> None:

    iris = load_iris(as_frame=True)

    data = iris.data
    data['target'] = iris.target

    # input_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    _ = await (
        DataImporters.literal(data)
        .process(
            [
                Transformations.ratio('petal length (cm)', 'sepal length (cm)', output='length petal / sepal'),
                Transformations.ratio('petal width (cm)', 'sepal width (cm)', output='width petal / sepal'),
                Transformations.ratio('petal length (cm)', 'sepal width (cm)', output='length / width petal / sepal'),
                Transformations.ratio('petal width (cm)', 'sepal length (cm)', output='width / length petal / sepal'),
                Transformations.exp_shift('width petal / sepal', 'width p/s exp'),
                Transformations.exp_shift('length petal / sepal', 'length p/s exp'),
            ]
        )
        .explore(
            [
                Explorers.histogram(target='target'),
                Explorers.correlation(),
                Explorers.pair_plot(
                    'target', features=['length p/s exp', 'width p/s exp', 'petal length (cm)', 'petal width (cm)']
                ),
            ]
        )
        .run()
    )
