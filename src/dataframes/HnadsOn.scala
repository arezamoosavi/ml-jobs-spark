package dataframes

import org.apache.spark.sql.SparkSession

object HnadsOn extends App {

  println("Hi, spark!")

  val spark = SparkSession.builder()
    .appName("Spark DataFrame App")
    .config("spark.master", "local")
    .getOrCreate()

  val df = spark.read.option("header", "true").option("inferSchema", "true").csv("src/dataframes/atm_data.csv")

  df.printSchema()
  df.show(10)
}
