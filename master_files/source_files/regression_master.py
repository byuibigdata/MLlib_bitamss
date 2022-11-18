# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #### Load Libraries, Data, and Define Evaluation Functions

# COMMAND ----------

# SKLearn Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

def get_scores(y_test, predictions):
    RMSE = mean_squared_error(y_test, predictions, squared=False)
    R2 = r2_score(y_test, predictions)

    print('RMSE:    ', int(RMSE))
    print('R2:      ', R2.round(4))

# COMMAND ----------

# MLlib Metrics
from pyspark.ml.evaluation import RegressionEvaluator

def mllib_metrics(predictions):
    rmse = RegressionEvaluator(labelCol="Scores", predictionCol="prediction", metricName="rmse")
    rmse = rmse.evaluate(predictions)
    print(rmse)

    r2 = RegressionEvaluator(labelCol="Scores", predictionCol="prediction", metricName="r2")
    r2 = r2.evaluate(predictions)
    print(r2)

# COMMAND ----------

# Again, we wanted to use data that we are familiar with from CSE 450.
df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv('dbfs:/FileStore/bitamss_mlib/score.csv')
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Data Preprocessing
# MAGIC ----
# MAGIC 
# MAGIC Features: Hours (Spent studying in hours)
# MAGIC 
# MAGIC Target: Scores (Grade received)
# MAGIC 
# MAGIC #### Splitting Datasets
# MAGIC 
# MAGIC [Documentation for Splitting Datasets using MLlib](https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html)
# MAGIC 
# MAGIC ##### Scaling, Normalizing, Bucketizing, etc.
# MAGIC 
# MAGIC [Documentation for Feature Extraction using MLlib](https://spark.apache.org/docs/1.4.1/ml-features.html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### MLlib

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

# 80% for training. 20% for testing.
train_data, test_data = df.randomSplit([0.8, 0.2])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### SKlearn

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler as MinMaxScalerSK
import pandas as pd

train_sk = train_data.toPandas()
test_sk = test_data.toPandas()

sc = MinMaxScalerSK()
train_sk[['Scaled_Hours']] = sc.fit_transform(train_sk[['Hours']])
test_sk[['Scaled_Hours']] = sc.transform(test_sk[['Hours']])

train_sk = pd.DataFrame(train_sk, columns = train_sk.columns)
test_sk = pd.DataFrame(test_sk, columns = test_sk.columns)

x_train_sk = train_sk[['Scaled_Hours']]
y_train_sk = train_sk[['Scores']]

x_test_sk = test_sk[['Scaled_Hours']]
y_test_sk = test_sk[['Scores']]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Gradient Boosted Regression

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### SKlearn

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
model_gb = GradientBoostingRegressor()
model_gb.fit(x_train_sk, y_train_sk)
predictions = model_gb.predict(x_test_sk)
get_scores(y_test_sk, predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### MLlib
# MAGIC 
# MAGIC [Documentation on Gradient Boosted Regression Models using MLlib](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression)

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

# Feature Extractions: https://spark.apache.org/docs/1.4.1/ml-features.html

# VectorAssembler Transformation - Converting column to vector type
vec_assembler = VectorAssembler(inputCols=['Hours'], outputCol="Hours_Vect")

# MinMaxScaler Transformation
scaler = MinMaxScaler(inputCol="Hours_Vect", outputCol="features")

# Model & Parameters
gbt = GBTRegressor(maxDepth=2, labelCol = 'Scores', featuresCol='features')

pipeline = Pipeline(stages=[vec_assembler, scaler, gbt])
model = pipeline.fit(train_data)
pred = model.transform(test_data)

# COMMAND ----------

mllib_metrics(pred)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Random Forest Regression

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### SKlearn

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor as RandomForestRegressorSK
model_rf = RandomForestRegressorSK()
model_rf.fit(x_train_sk, y_train_sk)
predictions = model_rf.predict(x_test_sk)
get_scores(y_test_sk, predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### MLlib
# MAGIC 
# MAGIC [Documentation on Random Forest Regression using MLlib](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-regression)

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# Feature Extractions: https://spark.apache.org/docs/1.4.1/ml-features.html

# VectorAssembler Transformation - Converting column to vector type
vec_assembler = VectorAssembler(inputCols=['Hours'], outputCol="Hours_Vect")

# MinMaxScaler Transformation
scaler = MinMaxScaler(inputCol="Hours_Vect", outputCol="features")

# Model & Parameters
rf = RandomForestRegressor(numTrees=5, maxDepth=2, labelCol = 'Scores', featuresCol='features')

pipeline_rf = Pipeline(stages=[vec_assembler, scaler, rf])
model_rf = pipeline_rf.fit(train_data)
pred_rf = model_rf.transform(test_data)

# COMMAND ----------

mllib_metrics(pred_rf)