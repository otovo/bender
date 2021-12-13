

from aioaws.s3 import S3Config
from pandas.core.frame import DataFrame
from bender.evaluator import Evaluator
from bender.importer import DataImporter
from bender.model_loader import ModelLoader
from bender.model_trainer import ModelTrainer
from bender.prediction_extractor import PredictionExtractor, PredictionOutput
from bender.split_strategy import SortedRatioSplitter, SplitStrategy
from bender.transformation import LogNormalDistributionShift, NeighbourDistance

class LeadQualifierExtractor(PredictionExtractor):

    @property
    def needed_features(self) -> set[str]:
        return set(['interest_id'])
        
    def output(self) -> tuple[str, PredictionOutput]:
        return ("probability", PredictionOutput.PROBABILITY)

    def extract(self, output: DataFrame):
        """Extract some information out of the predicted result

        Args:
            output (DataFrame): A data frame that contains all `needed_featues` and
                the predicted features
        """
        output['to_state'] = ['won'] * len(output)
        # then store the data frame in some way
        # this could be in Google Drive, sql database, etc.
        

async def test_run():

    s3_config = S3Config("", "", "", "")
    sql_url = "some url"
    sql_query = "some query"

    await (DataImporter
        .sql(sql_url, sql_query)
        .process([
            LogNormalDistributionShift("input_feature", "output_feature"),
            NeighbourDistance(number_of_neighbours=5),
        ])
        .load_model(
            ModelLoader.aws_s3("some/model.json", s3_config)
        )
        .predict(
            on=lambda df: df['some_var'] == 'some_state', 
            extractors=[LeadQualifierExtractor()]
        )
        .run()
        )
    
    model = await (
        DataImporter
        .sql(sql_url, sql_query)
        .process([
            LogNormalDistributionShift("input_feature", "output_feature")
        ])
        .split(SplitStrategy.ratio(0.7))
        .train(
            ModelTrainer.xgboost(), 
            input_features=['some_feature', 'more_features'], 
            target_feature='status'
        )
        .evaluate([
            Evaluator.precision_recall(),
            Evaluator.confusion_matrix(),
            Evaluator.roc_curve(),
        ])
        .run()
    )