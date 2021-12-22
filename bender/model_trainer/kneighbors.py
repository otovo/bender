from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

from sklearn.neighbors import KNeighborsClassifier

from bender.model_trainer.interface import ModelTrainer
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel
from bender.trained_model.kneighbours import TrainedKNeighboursClassifier


@dataclass
class KNeighborsConfig:
    n_neighbors: Optional[int] = None

    weights: Literal['uniform', 'distance'] = 'distance'

    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'auto'

    leaf_size: int = 30

    p: int = 2

    metric: str = 'minkowski'
    metric_params: Optional[dict[str, Any]] = None

    n_jobs: Optional[int] = None


class KNeighborsClassifierTrainer(ModelTrainer):

    config: KNeighborsConfig

    def __init__(self, config: KNeighborsConfig) -> None:
        self.config = config

    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        classifications = data_split.y_train.unique()
        dict_config = asdict(self.config)
        if 'n_neighbors' not in dict_config or dict_config['n_neighbors'] is None:
            dict_config['n_neighbors'] = len(classifications)
        model = KNeighborsClassifier(**dict_config)
        model.fit(data_split.x_train, data_split.y_train)
        return TrainedKNeighboursClassifier(model, data_split.x_features)
