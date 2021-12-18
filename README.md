# Bender 🤖

A Python package for faster, safer, and simpler ML processes.

## Why use `bender`?

Bender will make your machine learning processes, faster, safer, simpler while at the same time making it easy and flexible. This is done by providing a set base component, around the core processes that will take place in a ML pipeline process. While also helping you with type hints about what your next move could be.

## Pipeline Safety

The whole pipeline is build using generics from Python's typing system. Resulting in an improved developer experience, as the compiler can know if your pipeline's logic makes sense before it has started.

## Load a data set

Bender makes most of the `sklearn` datasets available through the `DataImporters.data_set(...)` importer. Here will you need to pass an enum to define which dataset you want. The following is shows how to do pre-processing, EDA, split, train, evaluate and return a metric that can be used to optimize the model for.

```python

await (DataImporters.data_set(DataSets.IRIS)
    .process([
        Transformations.exp_shift('petal length (cm)', output='pl exp'), # shift values with e^{feature}
        Transformations.exp_shift('petal width (cm)', output='pw exp'), # shift values with e^{feature}
    ])
    .explore([ # Do EDA's and understand the data
        Explorers.histogram(target='target'), # Display all features in a hist
        Explorers.correlation(input_features), # Display corr matrix and logs which features you could remove
        Explorers.pair_plot('target'), # View how features relate in 2D
    ])
    .split(SplitStrategies.uniform_ratio("target", 0.7)) # Have 70% of each target group in train and the rest in test
    .train(Trainers.kneighbours(), input_features=input_features, target_feature="target") # train kneighbours for each class
    .evaluate([
        Evaluators.confusion_matrix() # Only present the confusion matrix
    ])
    .metric(Metrics.log_loss()) # Compute the log loss
    .run())
```

## Training Example
Below is a simple example for training a XGBoosted tree
```python
DataImporters.sql(sql_url, sql_query) # Fetch SQL data

    .process([ # Preproces the data
        # Extract advanced information from json data
        Transformations.unpack_json("purchases", key="price", output_feature="price", policy=UnpackPolicy.median_number())

        Transformations.log_normal_shift("y_values", "y_log"),

        Transformations.date_component("month", "date", output_feature="month_value"), # Get date values from a date feature
    ])
    .split(SplitStrategies.ratio(0.7)) # Split 70 / 30% for train and test set

    .train( # Train a XGBoosted Tree model
        Trainers.xgboost(),
        input_features=['y_log', 'price', 'month_value', 'country', ...],
        target_feature='did_buy_product_x'
    )
    .evaluate([ # Evaluate how good the model is based on the test set
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
