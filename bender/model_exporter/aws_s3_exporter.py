from aioaws.s3 import S3Client, S3Config
from httpx import AsyncClient

from bender.model_exporter.interface import ModelExporter
from bender.trained_model.interface import TrainedModel


class AwsS3ModelExporter(ModelExporter):

    config: S3Client
    file_path: str

    def __init__(self, file_path: str, config: S3Config) -> None:
        self.file_path = file_path
        self.config = config

    async def export(self, model: TrainedModel) -> None:
        async with AsyncClient() as client:
            s3_clinet = S3Client(client, self.config)
            file_content = model.to_json()
            byte_data = bytes(file_content, 'utf-8')
            await s3_clinet.upload(self.file_path, byte_data)
