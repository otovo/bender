from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union

from sklearn.svm import SVR

from bender.model_trainer.interface import ModelTrainer
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel
from bender.trained_model.support_vector import SupportVectorRegression


class SvrKernal(Enum):
    LINEAR = 'linear'
    POLY = 'poly'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'
    PRECOMPUTED = 'precomputed'


GammaType = Union[Literal['auto', 'scale'], float]


@dataclass
class SvrTrainerConfig:
    kernal: SvrKernal
    degree: int = 3
    gamma: GammaType = 'scale'
    coef0: float = 0
    tol: float = 0.001
    C: float = 1
    elipson: float = 0.1
    shrinking: bool = True
    cache_size: float = 200
    verbose: bool = False
    max_iter: int = -1

    @staticmethod
    def linear(
        tol: float = 0.001,
        C: float = 1,
        elipson: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ) -> 'SvrTrainerConfig':
        return SvrTrainerConfig(
            kernal=SvrKernal.LINEAR,
            tol=tol,
            C=C,
            elipson=elipson,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    @staticmethod
    def poly(
        degree: int = 3,
        gamma: GammaType = 'scale',
        coef0: float = 1,
        tol: float = 0.001,
        C: float = 1,
        elipson: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ) -> 'SvrTrainerConfig':
        return SvrTrainerConfig(
            kernal=SvrKernal.LINEAR,
            tol=tol,
            C=C,
            elipson=elipson,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    @staticmethod
    def rbf(
        gamma: GammaType = 'scale',
        tol: float = 0.001,
        C: float = 1,
        elipson: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ) -> 'SvrTrainerConfig':
        return SvrTrainerConfig(
            gamma=gamma,
            kernal=SvrKernal.LINEAR,
            tol=tol,
            C=C,
            elipson=elipson,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    @staticmethod
    def sigmoid(
        gamma: GammaType = 'scale',
        coef0: float = 1,
        tol: float = 0.001,
        C: float = 1,
        elipson: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ) -> 'SvrTrainerConfig':
        return SvrTrainerConfig(
            kernal=SvrKernal.SIGMOID,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            elipson=elipson,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )


class SupportVectorRegressionTrainer(ModelTrainer):

    config: SvrTrainerConfig

    def __init__(self, config: SvrTrainerConfig) -> None:
        self.config = config

    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        model = SVR(
            kernel=self.config.kernal.value,
            degree=self.config.degree,
            gamma=self.config.gamma,
            coef0=self.config.coef0,
            tol=self.config.tol,
            C=self.config.C,
            epsilon=self.config.elipson,
            shrinking=self.config.shrinking,
            cache_size=self.config.cache_size,
            verbose=self.config.verbose,
            max_iter=self.config.max_iter,
        )
        model.fit(data_split.x_train, data_split.y_train)
        return SupportVectorRegression(model, data_split.x_features)
