from sklearn.tree import DecisionTreeClassifier

from bender.model_trainer.interface import ModelTrainer
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.decision_tree import TrainedDecisionTreeClassifier
from bender.trained_model.interface import TrainedModel


class DecisionTreeClassifierTrainer(ModelTrainer):
    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        model = DecisionTreeClassifier()
        model.fit(data_split.x_train, data_split.y_train)
        return TrainedDecisionTreeClassifier(model, data_split.x_features)
