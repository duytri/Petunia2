package main.scala

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import org.apache.spark.rdd.RDD

object TFIDFCalc {
  def tf(term: String, doc: Map[String, Int]): Double = {
    var wordCount = 0d
    doc.foreach((x => wordCount += x._2))
    doc(term) / wordCount
  }

  def idf(term: String, allDocs: Array[Map[String, Int]]): Double = {
    var n = 0d
    allDocs.foreach(x => {
      if (x.contains(term)) n += 1
    })

    return Math.log10(allDocs.length / n)
  }

  def tfIdf(word: (String, Int), doc: Map[String, Int], allDocs: Array[Map[String, Int]]): Double = {
    val term = word._1
    return tf(term, doc) * idf(term, allDocs)
  }
}