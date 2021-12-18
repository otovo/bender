from bender.model_trainer.kneighbors import KNeighborsClassifierTrainer
from bender.model_trainer.xgboosted_tree import XGBoostTrainer


class Trainers:
    @staticmethod
    def kneighbours() -> KNeighborsClassifierTrainer:
        return KNeighborsClassifierTrainer()

    @staticmethod
    def xgboost() -> XGBoostTrainer:
        return XGBoostTrainer()
