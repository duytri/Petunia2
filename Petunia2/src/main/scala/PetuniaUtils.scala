package main.scala

import scala.collection.mutable.Map
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer

object PetuniaUtils {
  def addOrIgnore(eachWordSet: Map[String, Int], someWords: ArrayBuffer[String]): Unit = {
    someWords.foreach { x =>
      {
        if (!eachWordSet.contains(x))
          eachWordSet += x -> 1
        else eachWordSet.update(x, eachWordSet(x) + 1)
      }
    }
  }

  def convert2RDD(rdd: RDD[LabeledPoint], label: LabeledPoint): RDD[LabeledPoint] = {
    rdd.map { x =>
      label
    }
  }
}