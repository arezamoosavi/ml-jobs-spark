package classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object TitanicApp {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder()
      .appName("Classification App")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    val data = spark.read.option("header", "true")
      .option("inferSchema", "true").format("csv")
      .load("src/classification/titanic.csv")

    data.printSchema()
    data.show(10)

    //    val colnames = data.columns
    //    val firstrow = data.head(1)(0)
    //    println("\n")
    //    println("Example Data Row")
    //    for (ind <- Range(1, colnames.length)) {
    //      println(colnames(ind))
    //      println(firstrow(ind))
    //      println("\n")
    //    }
    val logregdataall = data.select(data("Survived").as("label"), $"Pclass", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked")
    val logregdata = logregdataall.na.drop()
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

    val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
    val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

    // Assemble everything together to be ("label","features") format
    val assembler = (new VectorAssembler()
      .setInputCols(Array("Pclass", "SexVec", "Age","SibSp","Parch","Fare","EmbarkVec"))
      .setOutputCol("features") )

    val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
    val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(Array(genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,assembler, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(training)

    // Get Results on Test Set
    val results = model.transform(test)

    // Need to convert to RDD to use this
    val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

  }
}
