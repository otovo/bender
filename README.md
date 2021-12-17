# Bender 🤖

A Python package for faster, safer, and simpler ML processes.

## Why use `bender`?

Bender will make your machine learning processes, faster, safer, simpler while at the same time making it easy and flexible. This is done by providing a set base component, around the core processes that will take place in a ML pipeline process. While also helping you with type hints about what your next move could be.

## Pipeline Safety

The whole pipeline is build using generics from Python's typing system. Resulting in an improved developer experience, as the compiler can know if your pipeline's logic makes sense before it has started.

Bender will therefore make sure you **can't** make errors like

```python
# ⛔️ Invalid pipeline
DataImporters.sql(...)
    .process([...])
    # Compile Error: method `predict()` is not available
    .predict()

# ✅ Valid pipeline
DataImporters.sql(...)
    .process([...])
    .load_model(ModelLoader.aws_s3(...))
    .predict()
```

## Training Example
Below is a simple example for training a XGBoosted tree
```python
DataImporters
    # Fetch SQL data
    .sql(sql_url, sql_query)

    # Preproces the data
    .process([
        # Extract advanced information from json data
        Transformations.unpack_json("purchases", key="price", output_feature="price", policy=UnpackPolicy.median_number())

        Transformations.log_normal_shift("y_values", "y_log"),

        # Get date values from a date feature
        Transformations.date_component("month", "date", output_feature="month_value"),
    ])

    # Split 70 / 30% for train and test set
    .split(SplitStrategies.ratio(0.7))

    # Train a XGBoosted Tree model
    .train(
        ModelTrainer.xgboost(),
        input_features=['y_log', 'price', 'month_value', 'country', ...],
        target_feature='did_buy_product_x'
    )

    # Evaluate how good the model is based on the test set
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
