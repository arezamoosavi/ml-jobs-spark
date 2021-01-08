package regression

import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler

// To see less warnings
import org.apache.log4j._

object EcommerceModeling extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder()
    .appName("Regression App")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src/regression/Ecommerce Customers")

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

  val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")

  val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")


  // Use the assembler to transform our DataFrame to the two columns: label and features
  val output = assembler.setHandleInvalid("skip").transform(df.na.drop()).select($"label", $"features")


  // Create a Linear Regression Model object
  val lr = new LinearRegression()

  // Fit the model to the data and call this model lrModel
  val lrModel = lr.fit(output)

  // Print the coefficients and intercept for linear regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // Summarize the model over the training set and print out some metrics!
  // Use the .summary method off your model to create an object
  // called trainingSummary
  val trainingSummary = lrModel.summary

  // Show the residuals, the RMSE, the MSE, and the R^2 Values.
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"MSE: ${trainingSummary.meanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")

  println("SAVING THE MODEL ...")
  lrModel.write.overwrite().save("src/regression/ecommerce_linear_predictor.model")

  val loaded_model = LinearRegressionModel.load("src/regression/ecommerce_linear_predictor.model")

  println(f"The loss of loaded model: ${loaded_model.getLoss}")


}
