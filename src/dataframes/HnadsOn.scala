package dataframes

import org.apache.spark.sql.SparkSession
//import org.apache.spark.sql.functions.corr

import org.apache.spark.sql.functions._

object HnadsOn extends App {

  println("Hi, spark!")

  val spark = SparkSession.builder()
    .appName("Spark DataFrame App")
    .config("spark.master", "local")
    .getOrCreate()

  val df = spark.read.option("header", "true").option("inferSchema", "true").csv("src/dataframes/atm_data.csv")

  df.printSchema()
  df.describe().show()

  //  df.show(10)

  //  df.select("Amount_withdrawn_XYZ_Card").show(5)
  //  df.select(f"Amount_withdrawn_Other_Card", f"ATM_Name").show()

  import spark.implicits._

  //  df.filter($"Amount_withdrawn_Other_Card">1000000).show()

  //  df.select("Amount_withdrawn_Other_Card").filter("Amount_withdrawn_Other_Card < 100000").show()

  //  df.select("Amount_withdrawn_Other_Card", "Amount_withdrawn_XYZ_Card").filter("Amount_withdrawn_Other_Card < 100000 and Amount_withdrawn_XYZ_Card <100").show()

  //  df.select("Amount_withdrawn_Other_Card", "Amount_withdrawn_XYZ_Card")
  //    .filter($"Amount_withdrawn_Other_Card" < 1000000 && $"Amount_withdrawn_XYZ_Card" > 1000).show(10)

  //  val result_val = df.select("Amount_withdrawn_Other_Card", "Amount_withdrawn_XYZ_Card")
  //    .filter($"Amount_withdrawn_Other_Card" < 1000000 && $"Amount_withdrawn_XYZ_Card" > 1000).collect()

//  val number_of_results: Long = df.select("Amount_withdrawn_Other_Card", "Amount_withdrawn_XYZ_Card")
//    .filter($"Amount_withdrawn_Other_Card" < 1000000 && $"Amount_withdrawn_XYZ_Card" > 1000).count()

//  println(s"The number of filter results is ${number_of_results}\n\n")

//  df.filter($"Amount_withdrawn_Other_Card" ===139500).show()
//  df.filter("Amount_withdrawn_Other_Card=139500").show()

//  df.select(corr("Amount_withdrawn_Other_Card", "Amount_withdrawn_XYZ_Card")).show()

//  df.groupBy("ATM_Name").mean().show()
//  df.select(countDistinct("ATM_Name")).show()
//  df.select(collect_list("ATM_Name")).show()

  df.orderBy($"Transaction_Date".desc).show()
  df.select(year(col("Transaction_Date"))).orderBy($"Transaction_Date".desc).show()

}
