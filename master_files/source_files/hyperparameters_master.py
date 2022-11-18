# Databricks notebook source
# MAGIC %md
# MAGIC #### Hyperparameter Tuning for Gradient Boosted Regression

# COMMAND ----------

# MAGIC %md
# MAGIC Boosting is an iterative learning process which leverages computing power to improve the prediction accuracy of a set of weak learners. Within this use case, tree K's prediction outputs are weighed against K-1's outputs. Although decision trees alone are a classic example of supervised learning, as each target prediction Y is weighed against actual target value Y hat, ensemble methods like boosted trees utilize principles from reinforcement learning--if output K is a weak classifier relative to the set of trees as a whole, its outcomes will be weighed more heavily--in other words, ensemble methods are 'punished' for faulty outcomes to increase the predictive power of the entire model. In our use case, there are several hyperparameters we can tune to regulate overfitting, prevent underfitting, and increase overall model robustness.
# MAGIC 
# MAGIC We're using gradient boosted trees, which, in contrast to random forest, utilizes iterative, or sequential tree instantiation. While random forest builds trees stochastically (working via randomization to produce the best 'collective intelligence', similar to the approach genetic algorithms take), each gradient boosted tree K aims to minimize the gradient of the previous tree's loss function. Simply put, rather than picking the arbitrarily best-performing tree of the set, gradient boosting sequentially produces trees until the optimal tree is found with respect to the dataset and tree count. 
# MAGIC 
# MAGIC We will focus on a few prevalent hyperparameters:
# MAGIC 
# MAGIC 1 -> Max tree depth--prevents overfitting by disallowing individual trees to learn dynamics overly specific to single samples
# MAGIC 
# MAGIC 2 -> Minimum samples allowed per tree split iteration--reduces overfitting for the same reason as the previous hyperparameter--by keeping trees from learning overly specific dynamics
# MAGIC 
# MAGIC 3 -> Learning rate--while potentially computationally expensive, lowering the learning rate hyperparameter will increase the number of training iterations, maximizing generalization power for the model as a whole
# MAGIC 
# MAGIC 4 -> Iteration count--as a rule of thumb, more trees translates to a more versatile model. However, too many trees leads to overfitting, which will negatively impact the model's performance when evaluating data less homogenous to training data
# MAGIC 
# MAGIC 5 -> Loss function minimization--controls the 'punishment' the model receives for bad predictions; the value of this hyperparameter should be iteratively selected based on resultant classification accuracy
# MAGIC 
# MAGIC In the absence of nuanced understanding regarding one's chosen model and dataset, the prospect of manually editing hyperparameters is intimidating. Instead, we'll automate the process via grid search, a brute force mechanism which tries each possible tuning combination within a given range and returns the optimal set of hyperparameters. Let's begin!
# MAGIC 
# MAGIC First, we'll need to change our target name to 'label'--crossValidator requires this to work.

# COMMAND ----------

df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv('dbfs:/FileStore/bitamss_mlib/score.csv')
data = df.selectExpr("Hours as hours", "Scores as label")
display(data)

# COMMAND ----------

# Imports
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

# Assembler
vec_assembler = VectorAssembler(inputCols=['hours'], outputCol='hoursV')

# Split
train_data_tune, test_data_tune = data.randomSplit([0.8, 0.2])
tuned_gbt = GBTRegressor(labelCol='label', featuresCol='hoursV')

# COMMAND ----------

# MAGIC %md
# MAGIC Before we go any further, let's make sure the hyperparameters we want are available to us by calling .extractParamMap() on our gradient boosted tree object.

# COMMAND ----------

tuned_gbt.extractParamMap()

# COMMAND ----------

# Set pipeline
tuned_pipeline = Pipeline(stages=[vec_assembler, tuned_gbt])

# Define search space
gridSearch = (ParamGridBuilder()
             .addGrid(tuned_gbt.maxDepth, [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
             .addGrid(tuned_gbt.subsamplingRate, [.1, .3, .5, .7, .9])
             .addGrid(tuned_gbt.stepSize, [.1, .3, .5, .7, .9])
             .addGrid(tuned_gbt.maxIter, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125])
             .addGrid(tuned_gbt.lossType, ['squared', 'absolute'])
             .build())

# Build evaluator
gbtEval = RegressionEvaluator(labelCol = 'label', metricName='r2')
# Build crossvalidator
cv = CrossValidator(estimator=tuned_pipeline, estimatorParamMaps=gridSearch, evaluator=gbtEval, numFolds=3, parallelism=4)
tunedModel = cv.fit(train_data_tune)
tunedPred = tunedModel.transform(test_data_tune)

# COMMAND ----------

print('BestModel:\n\t-maxDepth =',tunedModel.bestModel._java_obj.getMaxDepth())
print('\t-numTrees =',tunedModel.bestModel._java_obj.getsubsamplingRate())
print('\t-stepSize =',tunedModel.bestModel._java_obj.getstepSize())
print('\t-maxIter =',tunedModel.bestModel._java_obj.getmaxIter())
print('\t-lossType =',tunedModel.bestModel._java_obj.getlossType())

# COMMAND ----------

print(gbtEval.evaluate(tunedPred))