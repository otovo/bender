from pandas.core.frame import DataFrame, Series
from bender.evaluators import Evaluators
from bender.importer import DataImporter
from bender.importers import DataImporters

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from bender.model_loaders import ModelLoaders
from bender.model_trainer import DecisionTreeClassifierTrainer, ModelTrainer
from bender.split_strategy import SplitStrategy
from bender.transformation import UnpackTypePolicy

from bender.transformations import Transformations

from sklearn.tree import DecisionTreeClassifier

pytestmark = pytest.mark.asyncio


async def test_predict_data():

    model = await (
        DataImporters.literal(
            DataFrame({
                'x': [0, 1],
                'y': [0, 1],
                'output': [0, 1]
            })
        )
        # No test set
        .split(SplitStrategy.ratio(1))
        .train(DecisionTreeClassifierTrainer(), input_features=['x', 'y'], target_feature='output')
        .run()
    )

    test_data = DataFrame({
        'x': [2, -3, 4],
        'y': [2, -3, 4]
    })
    expected = [1, 0, 1]
    result = await (
        ModelLoaders.literal(model)
            .import_data(DataImporters.literal(test_data))
            .predict()
            .run()
    )

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