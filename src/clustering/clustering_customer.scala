package clustering

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

// To see less warnings
import org.apache.log4j._

object clustering_customer extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder()
    .appName("Clustering App")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  val dataset: DataFrame = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load("src/clustering/Wholesale_Customers_Data.csv")

  dataset.printSchema()
  dataset.show(5)

  val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

  val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
  val training_data = assembler.transform(feature_data).select("features")
  val kmeans = new KMeans().setK(3).setSeed(1L)
  val model = kmeans.fit(training_data)

  // Make predictions
  val predictions = model.transform(training_data)

  // Evaluate clustering by computing Silhouette score
  val evaluator = new ClusteringEvaluator()

  val silhouette = evaluator.evaluate(predictions)
  println(s"Silhouette with squared euclidean distance = $silhouette")

  // Shows the result.
  println("Cluster Centers: ")
  val centers = model.clusterCenters
  centers.foreach(println)


}
