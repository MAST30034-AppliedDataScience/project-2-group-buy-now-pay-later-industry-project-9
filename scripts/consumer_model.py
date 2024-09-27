from pyspark.sql import functions as F, SparkSession
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
import pandas as pd



def consumer_model_dt(features, dataframe,rmse_evaluator, r2_evaluator):
    # VectorAssembler to combine the features into a single vector
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    # Prepare the data
    data = assembler.transform(dataframe)

    train_data, test_data = data.randomSplit([0.8, 0.2])  

    # Define model regressor
    dt = DecisionTreeRegressor(labelCol="average_fraud_probability", featuresCol="features")  
    
    # Parameter grid
    dt_param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [3, 5, 7]) \
        .addGrid(dt.maxBins, [32, 64]) \
        .build()
    
    # Cross-validation 
    dt_cv = CrossValidator(
    estimator=dt,
    estimatorParamMaps=dt_param_grid,
    evaluator=r2_evaluator,
    numFolds=3
    )

    # Pipeline 
    dt_pipeline = Pipeline(stages=[dt_cv])

    # Fit model

    # 6 mins
    dt_model = dt_pipeline.fit(train_data)

    # Make predictions on the test data
    dt_predictions = dt_model.transform(test_data)

    dt_rmse = rmse_evaluator.evaluate(dt_predictions)
    dt_r2 = r2_evaluator.evaluate(dt_predictions)
    print(f"Decision Tree RMSE: {dt_rmse}")
    print(f"Decision Tree R2: {dt_r2}")           # RMSE: 6.321439678625029 R2: 0.5426608139433955

    # Best model hyperparameters 
    best_dt_model = dt_model.stages[-1].bestModel
    print(f"Best Decision Tree maxDepth: {best_dt_model._java_obj.getMaxDepth()}")
    print(f"Best Decision Tree maxBins: {best_dt_model._java_obj.getMaxBins()}")

    #Feature Importances
    feature_names = assembler.getInputCols()
    best_dt_model = dt_model.stages[0].bestModel
    dt_feature_importances = best_dt_model.featureImportances
    dt_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": dt_feature_importances.toArray()
    }).sort_values(by="Importance", ascending=False)
    print(dt_importances_df)
    print()

    return


def consumer_model_rf(features, dataframe,rmse_evaluator, r2_evaluator):
    
    # VectorAssembler to combine the features into a single vector
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    # Prepare the data
    data = assembler.transform(dataframe)

    train_data, test_data = data.randomSplit([0.8, 0.2])
    
    # Define model regressor
    rf = RandomForestRegressor(labelCol="average_fraud_probability", featuresCol="features")

    # Parameter grid
    rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [5, 7]) \
    .build()

    # Cross-validation 
    rf_cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=rf_param_grid,
        evaluator=r2_evaluator,
        numFolds=3
    )

    rf_pipeline = Pipeline(stages=[rf_cv])

    # Fit the model
    rf_model = rf_pipeline.fit(train_data)

    rf_predictions = rf_model.transform(test_data)

    # Evaluation
    rf_rmse = rmse_evaluator.evaluate(rf_predictions)
    rf_r2 = r2_evaluator.evaluate(rf_predictions)
    print(f"Random Forest RMSE: {rf_rmse}")
    print(f"Random Forest R2: {rf_r2}")            # RMSE: 6.2324836442813565 R2: 0.5554417100291846


    # Best model hyperparameters
    best_rf_model = rf_model.stages[-1].bestModel
    print(f"Best Random Forest numTrees: {best_rf_model.getNumTrees}")
    print(f"Best Random Forest maxDepth: {best_rf_model.getMaxDepth()}")

    # Print Feature Importances
    feature_names = assembler.getInputCols()
    best_rf_model = rf_model.stages[0].bestModel
    rf_feature_importances = best_rf_model.featureImportances
    rf_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_feature_importances.toArray()
    }).sort_values(by="Importance", ascending=False)
    
    print(rf_importances_df)
    print()
    return


def consumer_model_lr(features, dataframe,rmse_evaluator, r2_evaluator):
    
    return