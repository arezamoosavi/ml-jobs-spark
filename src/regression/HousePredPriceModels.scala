package regression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.SparkSession

// To see less warnings
import org.apache.log4j._

object HousePredPriceModels extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder()
    .appName("Regression App")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src/regression/USA_Housing.csv")

  // Check out the Data
  data.printSchema()
  data.show(5)

  ////////////////////////////////////////////////////
  //// Setting Up DataFrame for Machine Learning ////
  //////////////////////////////////////////////////

  // Rename Price to label column for naming convention.
  // Grab only numerical columns from the data
  val df = data.select(data("Price").as("label"),
    $"Avg Area Income".cast("Double"), $"Avg Area House Age".cast("Double"), $"Avg Area Number of Rooms".cast("Double"), $"Area Population".cast("Double"))

  df.printSchema()
  df.show(5)

  // Set the input columns from which we are supposed to read the values
  // Set the name of the column where the vector will be stored
  val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income", "Avg Area House Age", "Avg Area Number of Rooms", "Area Population")).setOutputCol("features")

  // Use the assembler to transform our DataFrame to the two columns
  val output = assembler.transform(df).select($"label", $"features")
  output.show()

  // Create an array of the training and test data
  val Array(training, test) = output.select("label","features").randomSplit(Array(0.7, 0.3), seed = 12345)

  // Create a Linear Regression Model object
  val lr = new LinearRegression()

  /// PARAMETER GRID BUILDER //////////
  val paramGrid = new ParamGridBuilder().addGrid(lr.regParam,Array(1000,0.001)).build()

  // In this case the estimator is simply the linear regression.
  // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
  // 80% of the data will be used for training and the remaining 20% for validation.
  val evaluator = new RegressionEvaluator()
  val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(lr)
    .setEvaluator(evaluator.setMetricName("r2"))
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)


  // Note: Later we will see why we should split
  // the data first, but for now we will fit to all the data.
  val lrModel = trainValidationSplit.fit(training)

  // Summarize the model over the training set and print out some metrics!
  // Explore this in the spark-shell for more methods to call

  println(s"Metrics: ${lrModel.validationMetrics.toString}")

  lrModel.transform(test).select("features", "label", "prediction").show()

  println("SAVING THE MODEL ...")

  val model_path = "src/regression/house_linear_predictor_grids.model"
  lrModel.write.overwrite().save(model_path)
  val loaded_model = TrainValidationSplitModel.load(model_path)

  println(s"Metrics: ${lrModel.validationMetrics.toString}")
}
