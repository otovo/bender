from bender.factory_states import LoadedModel
from bender.model_loader import LiteralLoader, S3ModelLoader, S3Config
from bender.model_trainer import TrainedModel

class ModelLoaders:

    @staticmethod
    def aws_s3(file: str, config: S3Config) -> LoadedModel:
        return LoadedModel(S3ModelLoader(file, config))

    @staticmethod
    def literal(model: TrainedModel) -> LoadedModel:
        return LoadedModel(LiteralLoader(model))