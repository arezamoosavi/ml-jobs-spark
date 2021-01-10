package pca

import org.apache.spark.ml.feature.{PCA, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

// To see less warnings
import org.apache.log4j._

object pca_canser_analysis extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder()
    .appName("PCA App")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  val data: DataFrame = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load("src/pca/Cancer_Data")

  // Check out the Data
  data.printSchema()
  data.show(5)


  val colnames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
    "concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
    "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"))

  val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")

  // Use the assembler to transform our DataFrame to a single column: features
  val output = assembler.transform(data).select($"features")

  val scaler = (new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(false))

  // on the output of the VectorAssembler
  val scalerModel = scaler.fit(output)
  val scaledData = scalerModel.transform(output)

  // Then fit this to the scaledData
  val pca = (new PCA()
    .setInputCol("scaledFeatures")
    .setOutputCol("pcaFeatures")
    .setK(4)
    .fit(scaledData))

  // Call this new dataframe pcaDF
  val pcaDF = pca.transform(scaledData)

  // Show the new pcaFeatures
  val result = pcaDF.select("pcaFeatures")
  result.show()

}
