package classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object LogisticApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Classification App")
      .config("spark.master", "local")
      .getOrCreate()

    val training = spark.read.format("libsvm").load("src/classification/sample_libsvm_data.txt")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(training)
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    spark.stop()
  }

}
