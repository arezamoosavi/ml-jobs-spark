package regression

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


import org.apache.spark.sql.SparkSession

// To see less warnings
import org.apache.log4j._

object HousePredPrice extends App {
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
  //  CHECKING  the data:
  //  val colnames = data.columns
  //  val firstrow = data.head(1)(0)
  //  println("\n")
  //
  //  println("Example Data Row")
  //
  //  for (ind <- Range(1, colnames.length)) {
  //    println(colnames(ind))
  //    println(firstrow(ind))
  //    println("\n")
  //  }

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

  // Create a Linear Regression Model object
  val lr = new LinearRegression()

  // Fit the model to the data

  // Note: Later we will see why we should split
  // the data first, but for now we will fit to all the data.
  val lrModel = lr.fit(output)

  // Print the coefficients and intercept for linear regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // Summarize the model over the training set and print out some metrics!
  // Explore this in the spark-shell for more methods to call
  val trainingSummary = lrModel.summary

  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

  trainingSummary.residuals.show()

  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"MSE: ${trainingSummary.meanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")

  println("SAVING THE MODEL ...")
  lrModel.write.overwrite().save("src/regression/house_linear_predictor.model")

  val loaded_model = LinearRegressionModel.load("src/regression/house_linear_predictor.model")

  println(f"The loss of loaded model: ${loaded_model.getLoss}")



}
