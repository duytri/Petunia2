package main.scala

import scala.collection.mutable.Map
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.MapBuilder

object PetuniaUtils {
  def addOrIgnore(someWords: ArrayBuffer[String]): Map[String, Int] = {
    var eachWordSet = Map[String, Int]()
    someWords.foreach { x =>
      {
        if (!eachWordSet.contains(x))
          eachWordSet += x -> 1
        else eachWordSet.update(x, eachWordSet(x) + 1)
      }
    }
    eachWordSet
  }

  def convert2RDD(rdd: RDD[LabeledPoint], label: LabeledPoint): RDD[LabeledPoint] = {
    rdd.map { x =>
      label
    }
  }
}