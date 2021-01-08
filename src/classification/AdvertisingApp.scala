package classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.sql.functions.{col, hour, minute, second}

object AdvertisingApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder()
    .appName("Classification App")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  val data = spark.read.option("header", "true")
    .option("inferSchema", "true").format("csv")
    .load("src/classification/advertising.csv")

  data.printSchema()
  data.show(10)


  val timedata = data.withColumn("Hour", hour(data("Timestamp")))

  val logregdata = (timedata.select(data("Clicked on Ad").as("label"),
    $"Daily Time Spent on Site", $"Age", $"Area Income",
    $"Daily Internet Usage", $"Hour", $"Male"))

  val assembler = (new VectorAssembler()
    .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income",
      "Daily Internet Usage","Hour"))
    .setOutputCol("features") )


  val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

  val lr = new LogisticRegression()
  val pipeline = new Pipeline().setStages(Array(assembler, lr))

  val model = pipeline.fit(training)
  val results = model.transform(test)


  // Evaluations

  val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

  val metrics = new MulticlassMetrics(predictionAndLabels)

  println("Confusion matrix:")
  println(metrics.confusionMatrix)



}
