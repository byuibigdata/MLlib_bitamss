# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## MLlib Random Forest Classification

# COMMAND ----------

# import data and make it reaadable with pyspark
iris_df = spark.table('bitamss_mllib.iris')
display(iris_df)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

# string index the target
# assemble the feature into a vector

# Index features
# Fit on whole dataset to include all labels in index
string_indexer = StringIndexer(inputCol = """TODO: target col""",
                               outputCol = "species_indexed")

# Automatically identify categorical features, and index them
# Set maxCategories so features with > 4 distinct values are treated as continuous
vector_assembler = VectorAssembler(inputCols = ["""TODO: feature cols"""],
                                   outputCol = "features")

# Split the data into training and test sets
(trainingData, testData) = iris_df.randomSplit(["""TODO int % of training (ex: 0.75), TODO int % of testing (ex: 0.25)"""])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol = "species_indexed",
                            featuresCol = "features",
                            numTrees = """TODO: int num of trees (ex: 10)""")

# Chain indexers, vectorizers, and random forest in a Pipeline
pipeline = Pipeline(stages = [string_indexer, vector_assembler, rf])

# Run each step in the pipeline.
# Train model. This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol = "species_indexed",
                                              predictionCol = "prediction",
                                              metricName = "accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy(%):", accuracy)
print("Test Error:  %g" % (1.0 - accuracy))

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## SKLearn Random Forest Classification

# COMMAND ----------

iris_df = spark.table('bitamss_mllib.iris')
data = iris_df.toPandas()
display(data)

# COMMAND ----------

# Import train_test_split function
from sklearn.model_selection import train_test_split

X = data[["""TODO: features cols"""]]
y = data[["""TODO: target col"""]]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = """TODO: int % split for testing (ex: 0.3, 0.2)""")

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators = """TODO: int num of estimators (ex: 100, 80)""")

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# COMMAND ----------

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy(%):", metrics.accuracy_score(y_test, y_pred) * 100)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Sources:
# MAGIC 
# MAGIC - https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier
# MAGIC - https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
# MAGIC - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
# MAGIC - https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters
# MAGIC - https://www.kaggle.com/code/tcvieira/simple-random-forest-iris-dataset/notebook