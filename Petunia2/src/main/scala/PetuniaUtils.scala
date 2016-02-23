package main.scala

import scala.collection.mutable.Map
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ArrayBuffer
import java.io.File
import java.io.BufferedWriter
import java.io.FileWriter

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