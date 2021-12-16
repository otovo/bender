# type: ignore[misc]
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Optional

from pandas.core.frame import DataFrame
from pandas.core.series import Series

from bender.data_importer.importer import CachedImporter, DataImportable, DataImporter, JoinedImporter
from bender.evaluator.factory_method import Evaluable
from bender.evaluator.interface import Evaluator
from bender.model_exporter.factory import ModelExportable
from bender.model_exporter.interface import ModelExporter
from bender.model_loader.model_loader import ModelLoadable, ModelLoader
from bender.pipeline.interface import RunnablePipeline
from bender.prediction_extractor import Predictable, PredictionExtractor, PredictionOutput
from bender.split_strategy.split_strategy import Splitable, SplitStrategy, TrainingDataSet
from bender.trainer.model_trainer import ModelTrainer, Trainable, TrainedModel
from bender.transformation.transformation import Processable, Transformation

logger = logging.getLogger(__name__)


class LoadedDataAndModel(
    Processable,
    Predictable,
    RunnablePipeline[tuple[TrainedModel, DataFrame]],
):

    data_loader: LoadedData
    model_loader: LoadedModel

    def __init__(self, data_loader: LoadedData, model_loader: LoadedModel) -> None:
        self.data_loader = data_loader
        self.model_loader = model_loader

    def process(self, transformations: list[Transformation]) -> LoadedDataAndModel:
        # Some processing of the data
        return LoadedDataAndModel(self.data_loader.process(transformations), self.model_loader)

    def predict(self, on: Optional[Callable[[DataFrame], Series]] = None) -> PredictionPipeline:
        return PredictionPipeline(self, predict_on=on)

    async def run(self) -> tuple[TrainedModel, DataFrame]:
        processed_data = await self.data_loader.run()
        loaded_model = await self.model_loader.run()
        return (loaded_model, processed_data)


class PredictionPipeline(RunnablePipeline[Series]):

    data_and_model: LoadedDataAndModel
    predict_on: Optional[Callable[[DataFrame], Series]]

    def __init__(self, data_and_model: LoadedDataAndModel, predict_on: Optional[Callable[[DataFrame], Series]]) -> None:
        self.data_and_model = data_and_model
        self.predict_on = predict_on

    async def run(self) -> Series:
        model, processed_data = await self.data_and_model.run()

        if self.predict_on:
            predict_on_filter = self.predict_on(processed_data)
            predict_on_data = processed_data[predict_on_filter]
        else:
            predict_on_data = processed_data

        return model.predict(predict_on_data)


class ExtractFromPredictionPipeline(RunnablePipeline[None]):

    data_and_model: LoadedDataAndModel
    predict_on: Optional[Callable[[DataFrame], Series]]
    extractors: list[PredictionExtractor]

    def __init__(
        self,
        data_and_model: LoadedDataAndModel,
        predict_on: Callable[[DataFrame], Series],
        extractors: list[PredictionExtractor],
    ) -> None:
        self.data_and_model = data_and_model
        self.predict_on = predict_on
        self.extractors = extractors

    async def run(self) -> None:
        model, processed_data = await self.data_and_model.run()
        if self.predict_on:
            predict_on_filter = self.predict_on(processed_data)
            predict_on_data = processed_data[predict_on_filter]
        else:
            predict_on_data = processed_data

        for extractor in self.extractors:
            output_feature, output_type = extractor.output

            output_df = DataFrame()
            if output_type == PredictionOutput.CLASSIFICATION:
                output_df[output_feature] = model.predict(predict_on_data)
            elif output_type == PredictionOutput.PROBABILITY:
                output_df[output_feature] = model.predict_proba(predict_on_data)
            else:
                raise NotImplementedError()

            for needed_feature in extractor.needed_features:
                output_df[needed_feature] = predict_on_data[needed_feature]

            await extractor.extract(output_df)


class LoadedModel(DataImportable[LoadedDataAndModel], RunnablePipeline[TrainedModel]):

    model_loader: ModelLoader

    def __init__(self, model_loader: ModelLoader) -> None:
        self.model_loader = model_loader

    def import_data(self, importer: DataImporter) -> LoadedDataAndModel:
        return LoadedDataAndModel(importer, self)

    async def run(self) -> TrainedModel:
        return await self.model_loader.load_model()


class LoadedData(
    Processable,
    ModelLoadable[LoadedDataAndModel],
    RunnablePipeline[DataFrame],
    Splitable,
    DataImporter,
):

    importer: DataImporter
    transformations: list[Transformation]

    def __init__(self, importer: DataImporter, transformations: list[Transformation]) -> None:
        self.importer = importer
        self.transformations = transformations
        assert isinstance(importer, DataImporter)

    def process(self, transformations: list[Transformation]) -> LoadedData:
        # Some processing of the data
        return LoadedData(self.importer, self.transformations + transformations)

    def load_model(self, model_loader: ModelLoader) -> LoadedDataAndModel:
        return LoadedDataAndModel(self, LoadedModel(model_loader))

    async def import_data(self) -> DataFrame:
        return await self.run()

    async def run(self) -> DataFrame:
        logger.info('Fetching data')
        df = await self.importer.import_data()
        logger.info('Fetched data')

        for transformation in self.transformations:
            logger.info(f'Applying transformation {type(transformation)}')
            df = await transformation.transform(df)
            if df.empty:
                logger.error(f'DataFrame ended up as empty after a {type(transformation)}. Exiting early')
                raise Exception('Transformations returned an empty data set')
        return df

    def split(self, split_strategy: SplitStrategy) -> SplitedData:
        return SplitedData(self, split_strategy=split_strategy)

    def join_import(self, importer: DataImporter, join_key: str) -> LoadedData:
        return LoadedData(
            JoinedImporter(first_import=self.importer, second_import=importer, join_key=join_key), self.transformations
        )

    def cached(
        self, path: str, from_now: Optional[timedelta] = None, timestamp: Optional[datetime] = None
    ) -> LoadedData:
        importer: CachedImporter
        if timestamp:
            importer = CachedImporter(self.importer, path, timestamp)
        elif from_now:
            importer = CachedImporter(self.importer, path, datetime.now() + from_now)
        else:
            importer = CachedImporter(self.importer, path, datetime.now() + timedelta(days=1))
        return LoadedData(importer, self.transformations)


class SplitedData(RunnablePipeline[tuple[DataFrame, DataFrame]], Trainable['TrainingPipeline']):

    data_loader: LoadedData
    split_strategy: SplitStrategy

    def __init__(self, data_loader: LoadedData, split_strategy: SplitStrategy) -> None:
        self.data_loader = data_loader
        self.split_strategy = split_strategy

    async def run(self) -> tuple[DataFrame, DataFrame]:
        raw_data = await self.data_loader.run()
        return await self.split_strategy.split(raw_data)

    def train(self, model: ModelTrainer, input_features: list[str], target_feature: str) -> TrainingPipeline:
        return TrainingPipeline(self, model, input_features, target_feature)


class ValidationLoss(RunnablePipeline[float]):

    pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]]

    def __init__(self, pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]]) -> None:
        self.pipeline = pipeline

    async def run(self) -> float:
        model, data_set = await self.pipeline.run()
        return 0


class LossValidatable:
    def loss(self) -> ValidationLoss:
        raise NotImplementedError()


class ExportTrainedModel(RunnablePipeline[tuple[TrainedModel, TrainingDataSet]], LossValidatable):

    pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]]
    exporter: ModelExporter

    def __init__(
        self, pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]], exporter: ModelExporter
    ) -> None:
        self.pipeline = pipeline
        self.exporter = exporter

    async def run(self) -> tuple[TrainedModel, TrainingDataSet]:
        model, data_set = await self.pipeline.run()
        await self.exporter.export(model)
        return model, data_set

    def loss(self) -> ValidationLoss:
        return ValidationLoss(self)


class TrainAndEvaluatePipeline(
    RunnablePipeline[tuple[TrainedModel, TrainingDataSet]], ModelExportable[ExportTrainedModel], LossValidatable
):

    pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]]
    evaluators: list[Evaluator]

    def __init__(
        self, pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]], evaluators: list[Evaluator]
    ) -> None:
        self.pipeline = pipeline
        self.evaluators = evaluators

    async def run(self) -> tuple[TrainedModel, TrainingDataSet]:
        model, train_set = await self.pipeline.run()
        for evaluator in self.evaluators:
            await evaluator.evaluate(model, train_set)
        return model, train_set

    def export_model(self, exporter: ModelExporter) -> ExportTrainedModel:
        return ExportTrainedModel(self, exporter)

    def loss(self) -> ValidationLoss:
        return ValidationLoss(self)


class TrainingPipeline(
    RunnablePipeline[tuple[TrainedModel, TrainingDataSet]],
    Evaluable[TrainAndEvaluatePipeline],
    ModelExportable[ExportTrainedModel],
    LossValidatable,
):

    data_loader: SplitedData
    model_trainer: ModelTrainer
    input_features: list[str]
    target_feature: str

    def __init__(
        self, data_loader: SplitedData, model_trainer: ModelTrainer, input_features: list[str], target_feature: str
    ) -> None:
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.input_features = input_features
        self.target_feature = target_feature

    async def _training_set(self) -> TrainingDataSet:
        test_data, validation_data = await self.data_loader.run()
        return TrainingDataSet(self.input_features, self.target_feature, test_data, validation_data)

    async def run(self) -> tuple[TrainedModel, TrainingDataSet]:
        train_set = await self._training_set()
        model = await self.model_trainer.train(train_set)
        return model, train_set

    def evaluate(self, evaluators: list[Evaluator]) -> TrainAndEvaluatePipeline:
        return TrainAndEvaluatePipeline(self, evaluators)

    def export_model(self, exporter: ModelExporter) -> ExportTrainedModel:
        return ExportTrainedModel(self, exporter)

    def loss(self) -> ValidationLoss:
        return ValidationLoss(self)