from aioaws.s3 import S3Config

from bender.model_exporter.aws_s3_exporter import AwsS3ModelExporter
from bender.model_exporter.disk_exporter import DiskExporter


class ModelExporters:
    @staticmethod
    def aws_s3(file_path: str, config: S3Config) -> AwsS3ModelExporter:
        return AwsS3ModelExporter(file_path, config)

    @staticmethod
    def disk(file_path: str) -> DiskExporter:
        return DiskExporter(file_path)
