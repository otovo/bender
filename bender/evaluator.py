from __future__ import annotations
from typing import TypeVar

from bender.exporter import Exporter
from bender.model_trainer import TrainedModel, TrainedXGBoostModel
from bender.split_strategy import TrainingDataSet
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from xgboost import plot_importance

class Evaluator:

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet):
        """Evaluates a model

        Args:
            model ([type]): The model to evaluate
            data_set (DataSplit): The data that can be used to evaluate the model
        """
        raise NotImplementedError()


    # - staticmethods in order to discover evaluators


class RocCurve(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet):
        display = RocCurveDisplay.from_estimator(
            model.estimator(), 
            data_set.x_validate, 
            data_set.y_validate.astype(float)
        )
        _ = display.ax_.set_title("Roc Curve")
        await self.exporter.store_figure(display.figure_)

class CorrelationMatrix(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet):
        corr_heatmap = data_set.x_train.append(data_set.x_validate).corr()
        corr_threshold = 0.9
        for feature in data_set.x_features:
            is_feature_mask = corr_heatmap.columns == feature
            column_values = corr_heatmap[corr_heatmap.columns == feature]
            heatmap_mask = ((column_values > corr_threshold) | (column_values < -corr_threshold)) & (~is_feature_mask)
            mask = heatmap_mask.iloc[0]
            correlated_featrues = corr_heatmap.columns[mask]
            if len(correlated_featrues) == 0:
                continue
            new_mask = column_values.columns.isin(correlated_featrues)
            print("Warning: Correlated features should be considered to be removed")
            print(f"{feature} is related to {correlated_featrues}, corr values: {column_values.iloc[0][new_mask]}")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_heatmap, mask=np.zeros_like(corr_heatmap, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)
        ax.set_title("Correlation Matrix")
        await self.exporter.store_figure(fig)


class ConfusionMatrix(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet):
        display = ConfusionMatrixDisplay.from_estimator(
            model.estimator(), 
            data_set.x_validate, 
            data_set.y_validate.astype(float)
        )
        _ = display.ax_.set_title("Confusion Matrix")
        await self.exporter.store_figure(display.figure_)

class PredictProbability(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet):
        y_score = model.predict_proba(data_set.x_validate)[:,1]
        # Scores compared to true labels
        fig_hist = px.histogram(
            x=y_score, color=data_set.y_validate, nbins=50,
            labels=dict(color='True Label', x='Score',
            width=100, height=200)
        )
        await self.exporter.store_figure(fig_hist)

        # self.logger.report_plotly(title="p(x|y = 'lost' or 'won')",series='status',figure=fig_hist)


class XGBoostFeatureImportance(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet):
        if isinstance(model, TrainedXGBoostModel) is False:
            raise Exception("Only supporting feature importance for XGBoost models")
            
        fig, ax = plt.subplots(1,1, figsize=(20, 10))
        plot_importance(model.model, ax=ax)
        await self.exporter.store_figure(fig)

class PrecisionRecall(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet):
        display = PrecisionRecallDisplay.from_estimator(
            model.estimator(), 
            data_set.x_validate, 
            data_set.y_validate.astype(float)
        )
        _ = display.ax_.set_title("Precision-Recall curve")
        await self.exporter.store_figure(display.figure_)


EvaluableType = TypeVar('EvaluableType')

class Evaluable:

    def evaluate(self, evaluators: list[Evaluator]) -> EvaluableType:
        raise NotImplementedError()