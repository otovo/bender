from __future__ import annotations

import json
import logging
from typing import Any, Generic, TypeVar

from aioaws.s3 import S3Client, S3Config
from httpx import AsyncClient

from bender.trained_model.interface import TrainedModel
from bender.trained_model.xgboosted_tree import TrainedXGBoostModel

logger = logging.getLogger(__name__)


class ModelLoader:
    async def load_model(self) -> TrainedModel:
        raise NotImplementedError()

    @staticmethod
    def model_for(name: str) -> type[TrainedModel]:
        models = ModelLoader.supported_models()
        if name not in models:
            raise Exception(f'Unsupported model {name}')
        return models[name]

    @staticmethod
    def supported_models() -> dict[str, type[TrainedModel]]:
        return {'xgboost': TrainedXGBoostModel}


class LiteralLoader(ModelLoader):

    model: TrainedModel

    def __init__(self, model: TrainedModel) -> None:
        self.model = model

    async def load_model(self) -> TrainedModel:
        return self.model


class S3ModelLoader(ModelLoader):

    config: S3Config
    file_path: str

    def __init__(self, file_path: str, config: S3Config) -> None:
        self.file_path = file_path
        self.config = config

    async def load_model(self) -> TrainedModel:
        async with AsyncClient() as client:
            s3_client = S3Client(client, self.config)
            url = s3_client.signed_download_url(path=self.file_path)
            response = await client.get(url)
            logger.info(url)
            response.raise_for_status()
            body = response.content
            body_content: dict[str, Any] = json.loads(body)
            if 'name' not in body_content:
                raise Exception("Unsupported format on model. Missing 'name' parameter")
            model_type = S3ModelLoader.model_for(body_content['name'])
            return model_type.from_dict(body_content)


ModelLoaderType = TypeVar('ModelLoaderType')


class ModelLoadable(Generic[ModelLoaderType]):
    def load_model(self, model: ModelLoader) -> ModelLoaderType:
        raise NotImplementedError()
