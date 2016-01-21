package main.scala

import scala.collection.mutable.Map
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer

object PetuniaUtils {
  def addOrIgnore(eachWordSet: Map[String, Int], someWords: ArrayBuffer[String]): Unit = {
    val specialChars = Array((" "), ("."), (","), ("\t"), ("..."), ("#"), ("\u00a0"), ("("), (")"), ("-"), (":"))
    someWords.foreach { x =>
      {
        if (!specialChars.contains(x)) {
          val y = x.toLowerCase()
          if (!eachWordSet.contains(y))
            eachWordSet += y -> 1
          else eachWordSet.update(y, eachWordSet(y) + 1)
        }
      }
    }
  }
  
  def convert2RDD(rdd: RDD[LabeledPoint], label: LabeledPoint): RDD[LabeledPoint] = {
    rdd.map { x =>
      label
    }
  }
}