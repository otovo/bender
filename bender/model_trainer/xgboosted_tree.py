from typing import Any

from xgboost import XGBClassifier

from bender.model_trainer.interface import ModelTrainer
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel
from bender.trained_model.xgboosted_tree import TrainedXGBoostModel


class XGBoostTrainer(ModelTrainer):

    xgboost_parmas: dict[str, Any]

    def __init__(
        self,
        use_label_encoder: bool = False,
        learning_rate: float = 0.01,
        max_depth: int = 5,
        n_estimators: int = 400,
        verbosity: float = 0,
        scale_pos_weight: float = 1.0,
        gamma: float = 0,
        min_child_weight: float = 1,
        colsample_bytree: float = 1,
        reg_lambda: float = 1,
        alpha: float = 0,
    ) -> None:
        self.xgboost_parmas = {
            'use_label_encoder': use_label_encoder,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'verbosity': verbosity,
            'scale_pos_weight': scale_pos_weight,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'alpha': alpha,
        }

    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        # if data_split.y_train.dtype not in [int, bool, str]:
        #     print(data_split.y_train.dtypes)
        #     raise Exception('Training classification model on continuse values. Maybe you want a regression model?')
        model = XGBClassifier(**self.xgboost_parmas)
        # model.scale_pos_weight = int(np.round(data_split.x_train.shape[0] / data_split.y_train.sum() - 1))
        model.fit(data_split.x_train, data_split.y_train, eval_set=[(data_split.x_validate, data_split.y_validate)])
        return TrainedXGBoostModel(model, data_split.x_features)
