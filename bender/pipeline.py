from typing import List, Tuple, Any

from pandas.core.frame import DataFrame
from bender.evaluator import Evaluator
from bender.importer import DataImporter
from bender.model_trainer import ModelTrainer
from bender.split_strategy import DataSplit, SplitStrategy
from bender.transformation import SelectFeatures, Transformation

import logging

logger = logging.getLogger(__name__)

class Pipeline:

    importer: DataImporter
    pre_processing: List[Transformation]
    split_strategy: SplitStrategy
    input_features: List[str]
    model_to_train: ModelTrainer
    evaluators: List[Evaluator]

    def __init__(self, importer: DataImporter, pre_processing: List[Transformation], split_strategy: SplitStrategy, input_features: List[str], model: ModelTrainer, evaluators: List[Evaluator]) -> None:
        self.importer = importer
        self.pre_processing = pre_processing
        self.split_strategy = split_strategy
        self.model_to_train = model
        self.input_features = input_features
        self.evaluators = evaluators

    async def run(self) -> Tuple[Any, DataSplit]:
        logger.info("Fetching data")
        df = await self.importer.import_data()
        logger.info("Fetched data")

        for transformation in self.pre_processing:
            logger.info(f"Applying transformation {type(transformation)}")
            df = await transformation.transform(df)

        # Split data set
        logger.info(f"Splitting data with strategy: {type(self.split_strategy)}")
        data_set = await self.split_strategy.split(df)

        test_data_set = DataSplit(
            x_train=data_set.x_train[self.input_features],
            x_test=data_set.x_test[self.input_features],
            y_train=data_set.y_train,
            y_test=data_set.y_test
        )

        # Train model
        model = await self.model_to_train.train(test_data_set)

        # Evaluate model
        for evaluator in self.evaluators:
            await evaluator.evaluate(model, test_data_set)
        
        return (model, data_set)
        