# Bender 🤖

A package for getting you ML processes up and rolling.

## Why use `bender`?

Bender will make your machine learning processes, faster, safer, simpler while at the same time making it easy and flexible. This is done by providing a set base component, around the core processes that will take place in a ML pipeline process. While also helping you with type hints about what your next move could be.

## Example
Bellow is a simple example for training a XGBoosted tree
```python
DataImporters
    # Fetch SQL data
    .sql(sql_url, sql_query)

    # Preproces the data
    .process([
        # Extract advanced information from json data
        Transformations.unpack_json("purchases", key="price", output_feature="price", policy=UnpackTypePolicy.median_number())

        Transformations.log_normal_shift("y_values", "y_log"),
        
        # Get date values from a date feature
        Transformations.date_component("month", "date", output_feature="month_value"),
    ])

    # Split 70 / 30% for train and test set
    .split(SplitStrategies.ratio(0.7))
    
    # Train a XGBoosted Tree model
    .train(
        ModelTrainer.xgboost(), 
        input_features=['y_log', 'price', 'month_value', 'some_unhandled_feature', ...], 
        target_feature='did_buy_product_x'
    )

    # Evaluate how good the model is based on the test set
    .evaluate([
        Evaluators.roc_curve(),
        Evaluators.confusion_matrix(),
        Evaluators.precision_recall(),
    ])
```