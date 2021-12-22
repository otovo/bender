from bender.model_trainer.kneighbors import KNeighborsClassifierTrainer, KNeighborsConfig
from bender.model_trainer.linear_regression import LinearRegressionTrainer
from bender.model_trainer.support_vector import SupportVectorRegressionTrainer, SvrTrainerConfig
from bender.model_trainer.xgboosted_tree import XGBoostTrainer


class Trainers:
    @staticmethod
    def kneighbours(config: KNeighborsConfig = KNeighborsConfig()) -> KNeighborsClassifierTrainer:
        return KNeighborsClassifierTrainer(config)

    @staticmethod
    def xgboost() -> XGBoostTrainer:
        return XGBoostTrainer()

    @staticmethod
    def support_vector_regression(config: SvrTrainerConfig = SvrTrainerConfig.rbf()) -> SupportVectorRegressionTrainer:
        return SupportVectorRegressionTrainer(config)

    @staticmethod
    def linear_regression() -> LinearRegressionTrainer:
        return LinearRegressionTrainer()
