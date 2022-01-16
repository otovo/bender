# type: ignore[misc]
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from pandas.core.frame import DataFrame
from pandas.core.series import Series

from bender.data_exporter.disk import DiskDataExporter
from bender.data_exporter.interface import DataExporter
from bender.data_exporter.pipeline import Extractable
from bender.data_importer.importer import AppendImporter, CachedImporter, DataImportable, DataImporter, JoinedImporter
from bender.evaluator.factory_method import Evaluable
from bender.evaluator.interface import Evaluator
from bender.explorer.factory import Explorable
from bender.explorer.interface import Explorer
from bender.metric.factory import Metricable
from bender.metric.interface import Metric
from bender.model_exporter.factory import ModelExportable
from bender.model_exporter.interface import ModelExporter
from bender.model_loader.model_loader import ModelLoadable, ModelLoader
from bender.model_trainer.interface import ModelTrainer, Trainable
from bender.pipeline.interface import RunnablePipeline
from bender.prediction_extractor import Predictable, PredictionExtractor, PredictionOutput, ProbabilisticPredictable
from bender.split_strategy.split_strategy import Splitable, SplitStrategy, TrainingDataSet
from bender.trained_model.interface import TrainedModel, TrainedProbabilisticModel
from bender.transformation.transformation import Processable, Transformation

logger = logging.getLogger(__name__)


class ExplorePipeline(RunnablePipeline[DataFrame], Splitable):

    pipeline: RunnablePipeline[DataFrame]
    explorers: list[Explorer]

    def __init__(self, pipeline: RunnablePipeline[DataFrame], explorers: list[Explorer]) -> None:
        self.pipeline = pipeline
        self.explorers = explorers

    async def run(self) -> DataFrame:
        df = await self.pipeline.run()
        for explorer in self.explorers:
            await explorer.explore(df)
        return df

    def split(self, split_strategy: SplitStrategy) -> SplitedData:
        return SplitedData(self, split_strategy)


class LoadedDataAndModel(
    Processable,
    Predictable,
    ProbabilisticPredictable,
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

    def predict_proba(self, on: Optional[Callable[[DataFrame], Series]] = None) -> ProbalisticPredictionPipeline:
        return ProbalisticPredictionPipeline(self, predict_on=on)

    async def run(self) -> tuple[TrainedModel, DataFrame]:
        return await asyncio.gather(self.model_loader.run(), self.data_loader.run())


class Extractor(RunnablePipeline[DataFrame]):

    pipeline: RunnablePipeline[tuple[TrainedModel, DataFrame, Series]]
    prediction_feature: str
    features: list[str]
    exporter: DataExporter
    transformations: list[Transformation]

    def __init__(
        self,
        pipeline: RunnablePipeline[tuple[TrainedModel, DataFrame, Series]],
        prediction_feature: str,
        features: list[str],
        exporter: DataExporter,
        transformations: list[Transformation],
    ) -> None:
        self.pipeline = pipeline
        self.prediction_feature = prediction_feature
        self.features = features
        self.exporter = exporter
        self.transformations = transformations

    async def run(self) -> DataFrame:
        model, data, predictions = await self.pipeline.run()

        extraction = DataFrame()
        if self.features:
            extraction = data[self.features]
        extraction[self.prediction_feature] = predictions
        for transformation in self.transformations:
            extraction = await transformation.transform(extraction)
        await self.exporter.export(extraction)
        return data


class ExtractorProbability(RunnablePipeline[DataFrame]):

    pipeline: RunnablePipeline[tuple[TrainedModel, DataFrame, Series]]
    prediction_features: dict[Any, str]
    features: list[str]
    exporter: DataExporter
    transformations: list[Transformation]

    def __init__(
        self,
        pipeline: RunnablePipeline[tuple[TrainedModel, DataFrame, Series]],
        prediction_features: dict[Any, str],
        features: list[str],
        exporter: DataExporter,
        transformations: list[Transformation],
    ) -> None:
        self.pipeline = pipeline
        self.prediction_features = prediction_features
        self.features = features
        self.exporter = exporter
        self.transformations = transformations

    async def run(self) -> DataFrame:
        model, data, predictions = await self.pipeline.run()

        extraction = DataFrame()
        if self.features:
            extraction = data[self.features].copy()
        for classification, output_name in self.prediction_features.items():
            extraction[output_name] = predictions[classification].copy()
        for transformation in self.transformations:
            extraction = await transformation.transform(extraction)
        await self.exporter.export(extraction)
        return data


class PredictionPipeline(RunnablePipeline[tuple[TrainedModel, DataFrame, Series]], Extractable):

    data_and_model: LoadedDataAndModel
    predict_on: Optional[Callable[[DataFrame], Series]]

    def __init__(self, data_and_model: LoadedDataAndModel, predict_on: Optional[Callable[[DataFrame], Series]]) -> None:
        self.data_and_model = data_and_model
        self.predict_on = predict_on

    async def run(self) -> tuple[TrainedModel, DataFrame, Series]:
        model, processed_data = await self.data_and_model.run()

        if self.predict_on:
            predict_on_filter = self.predict_on(processed_data)
            predict_on_data = processed_data[predict_on_filter]
        else:
            predict_on_data = processed_data
        result = model.predict(predict_on_data)
        return (model, processed_data, result)

    def extract(
        self,
        prediction_as: str,
        metadata: Optional[list[str]] = None,
        exporter: DataExporter = DiskDataExporter('predictions.csv'),
        transforations: Optional[list[Transformation]] = None,
    ) -> Extractor:
        return Extractor(
            self,
            prediction_as,
            [] if metadata is None else metadata,
            exporter,
            [] if transforations is None else transforations,
        )


class ProbalisticPredictionPipeline(RunnablePipeline[tuple[TrainedModel, DataFrame, Series]]):

    data_and_model: RunnablePipeline[tuple[TrainedProbabilisticModel, DataFrame]]
    predict_on: Optional[Callable[[DataFrame], Series]]

    def __init__(
        self,
        data_and_model: RunnablePipeline[tuple[TrainedProbabilisticModel, DataFrame]],
        predict_on: Optional[Callable[[DataFrame], Series]],
    ) -> None:
        self.data_and_model = data_and_model
        self.predict_on = predict_on

    async def run(self) -> tuple[TrainedModel, DataFrame, Series]:
        model, processed_data = await self.data_and_model.run()

        if self.predict_on:
            predict_on_filter = self.predict_on(processed_data)
            predict_on_data = processed_data[predict_on_filter]
        else:
            predict_on_data = processed_data

        return (model, processed_data, model.predict_proba(predict_on_data))

    def extract(
        self,
        prediction_as: dict[Any, str],
        metadata: Optional[list[str]] = None,
        exporter: DataExporter = DiskDataExporter('predictions.csv'),
        transforations: Optional[list[Transformation]] = None,
    ) -> ExtractorProbability:
        return ExtractorProbability(
            self,
            prediction_as,
            [] if metadata is None else metadata,
            exporter,
            [] if transforations is None else transforations,
        )


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


class LoadedModel(DataImportable[LoadedDataAndModel], RunnablePipeline[TrainedModel], ModelLoader):

    model_loader: ModelLoader

    def __init__(self, model_loader: ModelLoader) -> None:
        self.model_loader = model_loader

    def import_data(self, importer: DataImporter) -> LoadedDataAndModel:
        return LoadedDataAndModel(importer, self)

    async def load_model(self) -> TrainedModel:
        return await self.run()

    async def run(self) -> TrainedModel:
        return await self.model_loader.load_model()


class LoadedData(
    Processable,
    ModelLoadable[LoadedDataAndModel],
    RunnablePipeline[DataFrame],
    Splitable,
    DataImporter,
    Explorable[ExplorePipeline],
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

    def append(self, importer: LoadedData, ignore_index: bool = True) -> LoadedData:
        """Append two differnet data importers.
        This can be usefull when you have different types of data, but with the same features

        Args:
            importer (DataImporter): The data to append

        Returns:
            DataImporter: A Importer that appends the multiple importers
        """
        return LoadedData(AppendImporter(self.importer, importer, ignore_index=ignore_index), self.transformations)

    def explore(self, explorers: list[Explorer]) -> ExplorePipeline:
        return ExplorePipeline(self, explorers)


class SplitedData(RunnablePipeline[tuple[DataFrame, DataFrame]], Trainable):

    data_loader: RunnablePipeline[DataFrame]
    split_strategy: SplitStrategy

    def __init__(self, data_loader: RunnablePipeline[DataFrame], split_strategy: SplitStrategy) -> None:
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


class PipelineMetric(RunnablePipeline[float]):

    metric: Metric
    pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]]

    def __init__(self, pipeline: RunnablePipeline[tuple[TrainedModel, TrainingDataSet]], metric: Metric) -> None:
        self.metric = metric
        self.pipeline = pipeline

    async def run(self) -> float:
        model, data_set = await self.pipeline.run()
        return await self.metric.metric(model, data_set)


class ExportTrainedModel(
    RunnablePipeline[tuple[TrainedModel, TrainingDataSet]], LossValidatable, Metricable[PipelineMetric]
):

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

    def metric(self, metric: Metric) -> PipelineMetric:
        return PipelineMetric(self, metric)


class TrainAndEvaluatePipeline(
    RunnablePipeline[tuple[TrainedModel, TrainingDataSet]],
    ModelExportable[ExportTrainedModel],
    LossValidatable,
    Metricable[PipelineMetric],
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
            logger.info(f'Evaluating with evaloator: {type(evaluator)}')
            await evaluator.evaluate(model, train_set)
        return model, train_set

    def export_model(self, exporter: ModelExporter) -> ExportTrainedModel:
        return ExportTrainedModel(self, exporter)

    def metric(self, metric: Metric) -> PipelineMetric:
        return PipelineMetric(self, metric)


class TrainingPipeline(
    RunnablePipeline[tuple[TrainedModel, TrainingDataSet]],
    Evaluable[TrainAndEvaluatePipeline],
    ModelExportable[ExportTrainedModel],
    LossValidatable,
    Metricable[PipelineMetric],
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

    def metric(self, metric: Metric) -> PipelineMetric:
        return PipelineMetric(self, metric)
