package main.scala

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

object TFIDFCalc {
  def tf(term: String, doc: HashMap[String, Int]): Double = {
    var wordCount = 0d
    doc.foreach((x => wordCount += x._2))
    doc(term) / wordCount
  }

  def idf(term: String, allDocs: Array[HashMap[String, Int]]): Double = {
    var n = 0d
    allDocs.foreach(x => {
      if (x.contains(term)) n += 1
    })

    return Math.log10(allDocs.length / n)
  }

  def tfIdf(word: (String, Int), docIndex: Int, allDocs: Array[HashMap[String, Int]]): Double = {
    val term = word._1
    val doc = allDocs(docIndex)
    return tf(term, doc) * idf(term, allDocs)
  }
  
  def idf2(term: String, allDocs: Array[HashMap[String, Int]]): (Int, Double) = {
    var n = 0d
    allDocs.foreach(x => {
      if (x.contains(term)) n += 1
    })

    return (allDocs.length -> n)
  }

  def tfIdf2(word: (String, Int), docIndex: Int, allDocs: Array[HashMap[String, Int]]): (Double, (Int, Double)) = {
    val term = word._1
    val doc = allDocs(docIndex)
    return (tf(term, doc) -> idf2(term, allDocs))
  }
}