package main.scala

import scala.collection.mutable.Map
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ArrayBuffer
import java.io.File
import java.io.BufferedWriter
import java.io.FileWriter
import org.apache.spark.rdd.RDD
import scala.io.Source

object PetuniaUtils {
  def addOrIgnore(someWords: ArrayBuffer[String]): Map[String, Int] = {
    var eachWordSet = Map[String, Int]()
    someWords.foreach { x =>
      {
        if (!eachWordSet.contains(x))
          eachWordSet += (x -> 1)
        else eachWordSet.update(x, eachWordSet(x) + 1)
      }
    }
    eachWordSet
  }

  def statWords(file: File): Seq[Map[String, Int]] = {
    var wordsTmpArr = new ArrayBuffer[String]
    val source = Source.fromFile(file.getAbsolutePath, "utf-8")
    source.getLines.foreach { y => wordsTmpArr.append(y) }
    // Fixed too many open files exception
    source.close
    Seq(addOrIgnore(wordsTmpArr))
  }

  def statTFIDF(doc: Map[String, Int], allDocs: Array[Map[String, Int]]): Seq[Map[String, Double]] = {
    var tfidfOneDoc = Map[String, Double]()
    doc.foreach(oneWord => {
      tfidfOneDoc += oneWord._1 -> TFIDFCalc.tfIdf(oneWord, doc, allDocs)
    })
    Seq(tfidfOneDoc)
  }

  def writeArray2File(array: ArrayBuffer[LabeledPoint], filePath: String): Unit = {
    val file = new File(filePath)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.flush()
    array.foreach { x =>
      var s = x.label + "\t"
      val dArray = x.features.toArray
      for (d <- dArray) {
        s += d + "\t"
      }
      s += "\n"
      bw.write(s)
    }
    bw.close()
  }

  def writeArray2File2(array: ArrayBuffer[String], filePath: String): Unit = {
    val file = new File(filePath)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.flush()
    array.foreach { x =>
      bw.write(x + "\n")
    }
    bw.close()
  }
}