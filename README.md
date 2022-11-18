# MLlib_bitamss

# Why use MLlib?

Spark provides a general machine learning library -- MLlib -- that is designed for simplicity, scalability, and easy integration with other tools. With the scalability, language compatibility, and speed of Spark, data scientists can solve and iterate through data problems faster. As can be seen in both the number of use cases and the large number of developer contributions.

#### Scalability
Spark provides data engineers and data scientists with a powerful, unified engine that is both fast and allows data engineers to solve their machine learning problems interactively and at much greater scale.
* Works very well with large datasets (GB --> PB)

#### Adaptable
* Compatable with Scala, Java, Python, and R

#### Simplicity
Simple APIs familiar to data scientists coming from tools like R and Python. Novices are able to run algorithms as beginners while experts can easily tune the system by adjusting paramters, so transitioning workflows is simple.

#### Speed
* Uses iterative computations
* Supports parallel computations

#### Uses
There are many but include
* Marketing
* Security
* Optimization

## What is MLlib?
On the databricks website it is defined as "Built on top of Spark, MLlib is a scalable machine learning library consisting of common learning algorithms and utilities, including classification, regression, clustering, collaborative filtering, dimensionality reduction, and underlying optimization primitives."

#### Databricks AutoML
* Automate machine learning
* Uses more of a "drag & drop" approach to coding aka "low code solution"
* Can be used for many different types of ML such as regression, classification and forecasting

According to Databricks on their ML Quickstart "You can visualize the different runs using a parallel coordinates plot, which shows the impact of different parameter values on a metric."

<img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/parallel-plot.png"/>

# When to use MLlib vs SKlearn

#### Comparison
Scikit-Learn has fantastic performance if your data fits into RAM. Python and Scikit-Learn do in-memory processing and in a non-distributed fashion. 

MLlib is not computationally efficient for small data sets when comparing with SKlearn (think of the dump truck analogy to drive to your friend's house), which is better off for small and medium sized data sets (megabytes, up to a few gigabytes). For much larger data sets, Spark ML really shines in comparison.

As data visualization is an integral part of machine learning, knowing that scikit-learn has support for Pandas and Matplotlib, makes the process of developing machine learning models very iterative and efficient. Results are easily visualized, assumptions verified, and Sklearn uses many scipy functions (such as normality tests or distribution fits) where required, as part of the machine learning workflow.

#### Recap

MLib...
* Can handle very large datasets
* Is very powerful
* Highly adaptable

SKlearn...
* Works well with small datasets (MB --> GB)
* Great for visualizations
* Uses in-memory processing

### Sources
* [Wiki](https://en.wikipedia.org/wiki/Federated_learning)
* [DataBricks MLlib](https://www.databricks.com/glossary/what-is-machine-learning-library#:~:text=Built%20on%20top%20of%20Spark,reduction%2C%20and%20underlying%20optimization%20primitives.)
* [AutoML](https://www.databricks.com/product/automl)
* [ML vs Sk](https://www.quora.com/How-is-scikit-learn-compared-with-Apache-Sparks-MLlib)
