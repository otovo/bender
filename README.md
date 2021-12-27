# Bender 🤖

A Python package for faster, safer, and simpler ML processes.

## Installation

`pip install benderml`

**Usage:**

```python
from bender.importers import DataImporters

pre_processed_data = await DataImporters.csv("file/to/data.csv").process([...]).run()
```

## Why use `bender`?

Bender will make your machine learning processes, faster, safer, simpler while at the same time making it easy and flexible. This is done by providing a set base component, around the core processes that will take place in a ML pipeline process. While also helping you with type hints about what your next move could be.

## Pipeline Safety

The whole pipeline is build using generics from Python's typing system. Resulting in an improved developer experience, as the compiler can know if your pipeline's logic makes sense before it has started.

## Load a data set

Bender makes most of the `sklearn` datasets available through the `DataImporters.data_set(...)` importer. Here will you need to pass an enum to define which dataset you want. It is also possible to load the data from sql, append different data sources and cache, and it is as simple as:
```python
from bender.importers import DataImporters

# Predifined data set
DataImporters.data_set(DataSets.IRIS)

# Load SQL
DataImporters.sql("url", "SELECT ...")

# Cache a sql import
DataImporters.sql("url", "SELECT ...")
    .cached("path/to/cache")
    .append(
        # Add more data from a different source (with same features)
        DataImporters.sql(...)
    )
```

## Processing
When the data has been loaded is usually the next set to process the data in some way. `bender` will therefore provide different components that transforms features. Therefore making it easier to keep your logic consistent over multiple projects.

```python
from bender.transformations import Transformations

DataImporters.data_set(DataSets.IRIS)
    .process([
        # pl exp = e^(petal length)
        Transformations.exp_shift('petal length (cm)', output='pl exp'),

        # Alternative to `exp_shift`
        Transformations.compute('pl exp', lambda df: np.exp(df['petal length (cm)'])),

        # purchases = mean value of the json price values
        Transformations.unpack_json("purchases", key="price", output_feature="price", policy=UnpackPolicy.median_number()),

        ...
    ])
```

## EDA

For view how the data is distribuated, is it also possible to explore the data.

```python
from bender.explorers import Explorers

await (DataImporters.data_set(DataSets.IRIS)
    .process([...])
    .explore([
        # Display all features in a hist
        Explorers.histogram(target='target'),

        # Display corr matrix and logs which features you could remove
        Explorers.correlation(input_features),

        # View how features relate in 2D
        Explorers.pair_plot('target'),
    ])
```

## Splitting into train and test sets
There are many ways we can train and test, it is therefore easy to choose and switch between how it is done with `bender`.

```python
from bender.split_strategies import SplitStrategies

await (DataImporters.data_set(DataSets.IRIS)
    .process([...])

    # Have 70% as train and 30 as test
    .split(SplitStrategies.ratio(0.7))

    # Have 70% of each target group in train and the rest in test
    .split(SplitStrategies.uniform_ratio("target", 0.7))

    # Sorts by the key and taks the first 70% as train
    .split(SplitStrategies.sorted_ratio("target", 0.7))
```

## Training
After you have split the data set into train and test, then you can train with the following.

```python
from bender.model_trainers import Trainers

await (DataImporters.data_set(DataSets.IRIS)
    .split(...)
    .train(
        # train kneighbours on the train test
        Trainers.kneighbours(),
        input_features=[...],
        target_feature="target"
    )
```

## Evaluate
After you have a model will it be smart to test how well it works.

```python
from bender.evaluators import Evaluators

await (DataImporters.data_set(DataSets.IRIS)
    .split(...)
    .train(...)
    .evaluate([
        # Only present the confusion matrix
        Evaluators.confusion_matrix(),
        Evaluators.roc_curve(),
        Evaluators.precision_recall(),
    ])
```

## Save model
At last would you need to store the model. You can therefore select one of manny exporters.
```python
from bender.exporters import Exporters

await (DataImporters.data_set(DataSets.IRIS)
    .split(...)
    .train(...)
    .export_model(Exporters.aws_s3(...))
```

## Predict
```python
ModelLoaders
    .aws_s3("path/to/model", s3_config)
    .import_data(
        DataImporters.sql(sql_url, sql_query)
    )
    .predict()
```

## Extract result
```python
ModelLoaders
    .aws_s3(...)
    .import_data(...)
    .predict()
    .extract(prediction_as="target", metadata=['entry_id'], exporter=DataExporters.disk("predictions.csv"))
```

## Examples
An example of the IRIS data set which trains a model to perfection

```python
await (DataImporters.data_set(DataSets.IRIS)
    .process([
        Transformations.exp_shift('petal length (cm)', output='pl exp'),
        Transformations.exp_shift('petal width (cm)', output='pw exp'),
    ])
    .explore([
        Explorers.histogram(target='target'),
        Explorers.correlation(input_features),
        Explorers.pair_plot('target'),
    ])
    .split(SplitStrategies.uniform_ratio("target", 0.7))
    .train(Trainers.kneighbours(), input_features=input_features, target_feature="target")
    .evaluate([
        Evaluators.confusion_matrix()
    ])
    .metric(Metrics.log_loss())
    .run())
```

## XGBoost Example
Below is a simple example for training a XGBoosted tree
```python
DataImporters.sql(sql_url, sql_query)

    .process([ # Preproces the data
        # Extract advanced information from json data
        Transformations.unpack_json("purchases", key="price", output_feature="price", policy=UnpackPolicy.median_number())

        Transformations.log_normal_shift("y_values", "y_log"),

        # Get date values from a date feature
        Transformations.date_component("month", "date", output_feature="month_value"),
    ])
    .split(SplitStrategies.ratio(0.7))

    # Train a XGBoosted Tree model
    .train(
        Trainers.xgboost(),
        input_features=['y_log', 'price', 'month_value', 'country', ...],
        target_feature='did_buy_product_x'
    )
    .evaluate([
        Evaluators.roc_curve(),
        Evaluators.confusion_matrix(),
        Evaluators.precision_recall(
            # Overwrite where to export the evaluated result
            Exporter.disk("precision-recall.png")
        ),
    ])
```

## Predicting Example

Below will a model be loaded from a AWS S3 bucket, preprocess the data, and predict the output.
This will also make sure that the features are valid before predicting.

```python
ModelLoaders
    # Fetch Model
    .aws_s3("path/to/model", s3_config)

    # Load data
    .import_data(
        DataImporters.sql(sql_url, sql_query)
            # Caching import localy for 1 day
            .cached("cache/path")
    )
    # Preproces the data
    .process([
        Transformations.unpack_json(...),
        ...
    ])
    # Predict the values
    .predict()
```
