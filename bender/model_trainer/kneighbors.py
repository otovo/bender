from sklearn.neighbors import KNeighborsClassifier

from bender.model_trainer.interface import ModelTrainer
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel
from bender.trained_model.kneighbours import TrainedKNeighboursClassifier


class KNeighborsClassifierTrainer(ModelTrainer):
    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        classifications = data_split.y_train.unique()
        model = KNeighborsClassifier(n_neighbors=len(classifications))
        model.fit(data_split.x_train, data_split.y_train)
        return TrainedKNeighboursClassifier(model, data_split.x_features)
