from datetime import datetime
from logging import log
from typing import Optional
from bender.pipeline import Pipeline
from bender.importer import PostgreSQLImporter
from bender.exporter import LocalDiskExporter
from bender.split_strategy import RandomRatioSplitter, SortedRatioSplitter
from bender.transformation import DropNaN, LogNormalDistributionShift, Filter, DateComponent, RemoveDuplicateColumns, DescribeData, LogDataHead, LogDataInfo, SelectFeatures, DiscardFeatures, UnpackJson, UnpackList, BinaryTransform, NeighbourDistance, CastColumn
from bender.model_trainer import XGBoostTrainer
from bender.evaluator import PredictProbability, RocCurve, ConfusionMatrix, XGBoostFeatureImportance
import asyncio
from logging.config import dictConfig
from pydantic import BaseSettings, PostgresDsn
from pandas import Timestamp

cloud_db_query = """
WITH interest_calculation_cte AS (
                    SELECT DISTINCT ON (ii.id)
                        ii.id AS interest_id,
                        ii.address_id AS address_id,
                        la.point AS address_point,
                        la.latitude AS latitude,
                        la.longitude AS longitude,
                        la.country AS country,
                        ii.preferred_surface_id AS interest_preferred_surface_id,
                        ii.created_at AS interest_created_at,
                        ii.status AS interest_status,
                        ic.id AS interest_calc_id,
                        ii.business_unit_id AS bu_id,
                        ii.agent_id AS agent_id,
                        ii.utm_data AS utm_data,
                        ic.created_at AS interest_calc_created_at
                    FROM interests_interest ii
                        JOIN interests_calculation ic ON ii.id = ic.interest_id
                        JOIN location_address la ON ii.address_id = la.id
                    WHERE ii.created_at BETWEEN ('2021-10-01' :: TIMESTAMP AT TIME ZONE 'Europe/Oslo') AND ('2021-10-30' :: TIMESTAMP AT TIME ZONE 'Europe/Oslo')
                    ORDER BY 1, ic.created_at DESC
                ),

                pricing_meta_cte AS (
                    SELECT
                        icd.interest_id,
                        JSON_AGG(
                            JSON_BUILD_OBJECT(
                                'package_id', ipp.id,
                                'package_name', ipp.package,
                                'price_currency', peps.gross_price_currency,
                                'price_value', peps.gross_price
                            )
                        ) AS interest_package_meta
                    FROM interest_calculation_cte icd
                        LEFT JOIN interests_packageprice AS ipp ON ipp.calculation_id = icd.interest_calc_id
                        LEFT JOIN priceengine_pricespec AS peps ON ipp.price_id = peps.id
                    GROUP BY 1
                ),

                surface_data_cte AS (
                    SELECT DISTINCT ON (ac.address_id)
                        ss.address_id,
                        ss.id AS surface_id,
                        ss.area AS surface_area,
                        ss.azimuth AS surface_azimuth,
                        ss.slope AS surface_slope,
                        ss.polygon AS surface_polygon,
                        ss.circumference AS surface_circumference,
                        ss.max_num_panels AS surface_max_num_panels,
                        ss.score AS surface_score,
                        ss.aerial_photo AS surface_aerial_photo,
                        ss.eave_height AS surface_eave_height,
                        ss.created_at AS surface_created_at,
                        ss.updated_at AS surface_updated_at,
                        ss.energy_yield AS surface_energy_yield,
                        ss.is_outline AS surface_is_outline,
                        ss.building_id AS surface_building_id,
                        ss.is_fake AS surface_is_fake
                    FROM interest_calculation_cte ac
                        JOIN surfaces_surface ss USING (interest_id)
                    ORDER BY ac.address_id, ss.energy_yield DESC
                ),

                address_count_cte AS (
                    SELECT
                        address_id,
                        COUNT(*) AS address_id_count
                    FROM interest_calculation_cte
                    GROUP BY 1
                )

            SELECT
                ic.interest_id,
                ic.interest_calc_id,
                ic.interest_status,
                ic.address_id,
                ic.bu_id,
                ic.utm_data,
                ic.agent_id,
                ic.address_point,
                ic.latitude,
                ic.longitude,
                ic.country,
                -- 1. Time of entry into our site (interest creation time)
                ic.interest_created_at,
                -- 2. Number of times this particular address has been queried
                ac.address_id_count AS address_query_count,
                -- 3. Power properties of the preferred surface in the calculation (Azimuth, Slope, irradiation)
                sd.*,
                -- interest pricing data
                pm.interest_package_meta
            FROM interest_calculation_cte ic
                LEFT JOIN surface_data_cte sd USING (address_id)
                LEFT JOIN pricing_meta_cte pm USING (interest_id)
                JOIN address_count_cte ac USING (address_id)
                ;
"""

country: Optional[str] = None

pre_activation_phase = False
logic = "I -> O"

# pipeline_transformations = [
    
#     RemoveDuplicateColumns(),

#     Filter(lambda df: df["country"] == country)
#         .only_if(country is not None),

#     UnpackJson("interest_package_meta", key="price_value", output_feature="price_package_1"),

#     DateComponent("dayofweek", "interest_created_at", output_feature="created_weekday"),
#     DateComponent("hour", "interest_created_at", output_feature="created_hour_of_day"),

#     NeighbourDistance(number_of_neighbours=3, to=lambda df: df["interest_status"] == "won"),

#     LogNormalDistributionShift("surface_area", output_feature="surface_area"),
#     LogNormalDistributionShift("surface_circumference", output_feature="surface_circumference"),
# ]

# if pre_activation_phase:
#     pipeline_transformations += [
#         Filter(lambda df: df['interest_status'].isin(['won','lost'])),
#         BinaryTransform("status", lambda df: df["interest_status"] == "won")
#     ]
# else:
#     if logic == "I -> O":
#         pipeline_transformations += [
#             Filter(lambda df: df['interest_status'].isin(['won', 'calc_accepted', 'offer_created', 'offer_accepted'])),
#             BinaryTransform("status", lambda df: df["interest_status"] != "calc_accepted")
#         ]
#     else:
#         pipeline_transformations += [
#             Filter(lambda df: df['interest_status'].isin(['offer_created', 'offer_accepted'])),
#             BinaryTransform("status", lambda df: df["interest_status"] != "offer_created")
#         ]

split_date = Timestamp(datetime(year=2021, month=10, day=15)).tz_localize("utc")

class Settings(BaseSettings):
    cloud_database_url: PostgresDsn

    class Config:
        env_file = ".env"

settings = Settings()

pipeline = Pipeline(
    importer=(
        PostgreSQLImporter(url=settings.cloud_database_url, query=cloud_db_query)
            .cached("./data/cloud")
    ),

    pre_processing=[
    
        RemoveDuplicateColumns(),

        Filter(lambda df: 
            df["country"] == country)
                .only_if(country is not None),

        UnpackJson("interest_package_meta", key="price_value", output_feature="price_package_1", data_type=float),
        UnpackJson("utm_data", key="utm_source", output_feature="utm_source", data_type=str),

        DateComponent("dayofweek", "interest_created_at", output_feature="created_weekday"),
        DateComponent("hour", "interest_created_at", output_feature="created_hour_of_day"),

        NeighbourDistance(number_of_neighbours=3, to=lambda df: 
            (df["interest_status"] == "won") & (df["interest_created_at"] < split_date)),

        LogNormalDistributionShift("surface_area", output_feature="surface_area"),
        LogNormalDistributionShift("surface_circumference", output_feature="surface_circumference"),

        Filter(lambda df: 
            df['interest_status'].isin(['won', 'calc_accepted', 'offer_created', 'offer_accepted'])),

        BinaryTransform("status", lambda df: 
            df["interest_status"] != "calc_accepted"),

        LogDataInfo()
    ],
    
    split_strategy=SortedRatioSplitter(
        ratio=0.8, 
        sort_key="interest_created_at", 
        y_feature="status"
    ),

    input_features=[
        "address_query_count",
        "surface_area",
        "surface_azimuth",
        "surface_slope",
        "surface_circumference",
        "surface_max_num_panels",
        "surface_score",
        "surface_eave_height",
        "surface_energy_yield",
        "created_weekday",
        "created_hour_of_day",
        "neighbor_distance_1",
        "neighbor_distance_2",
        "neighbor_distance_3",
        "price_package_1"
    ],
    
    model=XGBoostTrainer(),

    evaluators=[
        RocCurve(LocalDiskExporter("./data/roc-curve")),
        ConfusionMatrix(LocalDiskExporter("./data/confusion-matrix")),
        PredictProbability(LocalDiskExporter("./data/predict-prob")),
        XGBoostFeatureImportance(LocalDiskExporter("./data/xgb-feature-importance"))
    ]
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {},
    "formatters": {
        "simple": {
            "class": "logging.Formatter",
            "datefmt": "%H:%M:%S",
            "format": "%(levelname)s %(asctime)s %(name)s:%(lineno)d %(message)s",
        },
    },
    "handlers": {
        "simple": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "loggers": {
        "": {
            "handlers": ["simple"],
            "level": "INFO",
            "propagate": True,
        },
        "databases": {
            "handlers": ["simple"],
            "level": "INFO",
        },
    },
}
dictConfig(LOGGING)  # Configures logging based on ^  # Configures logging based on ^



async def main():
    # Mapping the trained pickels to the name used in this repo

    result = await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())