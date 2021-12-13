from bender.factory_states import LoadedModel
from bender.model_loader import S3ModelLoader, S3Config

class ModelLoaders:

    @staticmethod
    def aws_s3(file: str, config: S3Config) -> LoadedModel:
        return LoadedModel(S3ModelLoader(file, config))