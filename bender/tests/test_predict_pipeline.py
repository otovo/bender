import numpy as np
import pytest
from pandas.core.frame import DataFrame

from bender.importers import DataImporters
from bender.model_loaders import ModelLoaders
from bender.model_trainer.decision_tree import DecisionTreeClassifierTrainer
from bender.split_strategies import SplitStrategies

pytestmark = pytest.mark.asyncio


async def test_predict_data() -> None:

    model, data_set = await (
        DataImporters.literal(DataFrame({'x': [0, 1], 'y': [0, 1], 'output': [0, 1]}))
        # No test set
        .split(SplitStrategies.ratio(1))
        .train(DecisionTreeClassifierTrainer(), input_features=['x', 'y'], target_feature='output')
        .run()
    )

    test_data = DataFrame({'x': [2, -3, 4], 'y': [2, -3, 4]})
    expected = [1, 0, 1]
    _, _, result = await (ModelLoaders.literal(model).import_data(DataImporters.literal(test_data)).predict().run())

    assert np.all(expected == result)

    """
    Supervised Regression

    Vector[float] -> float

    .train(
        RegresionModels.linear(),
        input_features=["area", "location"], # floats
        target_feature="price" # float
    )
    """

    """
    Supervised Classification

    Vector[float / int / bool / str] -> str / bool / int

    .train(
        ClassificationModels.DecisionTree(),
        input_features=["sepal_length", "sepal_width"], # float / int / bool / str
        target_feature="class_name" # str / bool / int
    )

    # Should only be avaialbe for clustering / classification problems
    .predict_probability(
        labels={
            "setosa": "is_setosa_probability",
            "versicolor": "is_versicolor_probability",
        }
    )
    """
