package main.scala

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.Map
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.util.Properties
import java.io.File
import java.io.BufferedWriter
import java.io.FileWriter

object PetuniaMain {
  def main(args: Array[String]): Unit = {
    //~~~~~~~~~~Initialization~~~~~~~~~~
    val conf = new SparkConf().setAppName("ISLab.Petunia").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val currentDir = new File(".").getCanonicalPath
    println(currentDir)
    //~~~~~~~~~~Get all data directories~~~~~~~~~~

    val inputDirPath = currentDir + "/data/in"

    val input0 = inputDirPath + File.separator + "0"
    val input1 = inputDirPath + File.separator + "1"

    val inputDirFile0 = new File(input0)
    val inputDirFile1 = new File(input1)

    //~~~~~~~~~~Get all input files~~~~~~~~~~
    val listFiles0 = inputDirFile0.listFiles
    val listFiles1 = inputDirFile1.listFiles
    var inputFiles = listFiles0 ++ listFiles1

    var wordSetByFile = new ArrayBuffer[Map[String, Int]](inputFiles.length) // Map[word, frequency in document]
    //Foreach text file
    for (i <- 0 to inputFiles.length - 1) {
      var wordsTmpArr = new ArrayBuffer[String]
      val source = Source.fromFile(inputFiles(i).getAbsolutePath, "utf-8")
      source.getLines.foreach { x => wordsTmpArr.append(x) }
      // Fixed too many open files exception
      source.close
      wordSetByFile.append(PetuniaUtils.addOrIgnore(wordsTmpArr))
    }
    println("inputDirFile0: " + listFiles0.length)
    println("inputDirFile1: " + listFiles1.length)
    println("inputFiles: " + inputFiles.length)
    //~~~~~~~~~~Calculate TFIDF~~~~~~~~~~
    var tfidfWordSet = new ArrayBuffer[Map[String, Double]](inputFiles.length) // Map[word, TF*IDF-value]
    for (i <- 0 to inputFiles.length - 1) {
      for (oneWord <- wordSetByFile(i)) {
        tfidfWordSet.append(Map(oneWord._1 -> TFIDFCalc.tfIdf(oneWord, i, wordSetByFile)))
      }
    }

    //~~~~~~~~~~Remove stopwords~~~~~~~~~~
    //// Load stopwords from file
    val stopwordFilePath = "./libs/vietnamese-stopwords.txt"
    var arrStopwords = new ArrayBuffer[String]
    val swSource = Source.fromFile(stopwordFilePath)
    swSource.getLines.foreach { x => arrStopwords.append(x) }
    swSource.close
    //// Foreach document, remove stopwords
    for (i <- 0 to inputFiles.length - 1) {
      tfidfWordSet(i) --= arrStopwords
    }

    //~~~~~~~~~~Normalize by TFIDF~~~~~~~~~~
    val lowerUpperBound = (0.00135d, 0.7d)
    var attrWords = ArrayBuffer[String]()
    for (i <- 0 to inputFiles.length - 1) {
      tfidfWordSet(i).foreach(x => {
        if (x._2 <= lowerUpperBound._1 || x._2 >= lowerUpperBound._2) {
          tfidfWordSet(i).remove(x._1)
        } else attrWords += x._1
      })
    }
    PetuniaUtils.writeArray2File2(attrWords, "./libs/attr.txt")

    //~~~~~~~~~~Create vector~~~~~~~~~~
    var vectorWords = new ArrayBuffer[LabeledPoint]
    for (i <- 0 to inputFiles.length - 1) {
      var vector = new ArrayBuffer[Double]
      for (word <- attrWords) {
        if (tfidfWordSet(i).contains(word)) {
          vector.append(tfidfWordSet(i).get(word).get)
        } else vector.append(0d)
      }
      if (i < listFiles0.length) {
        vectorWords.append(LabeledPoint(0d, Vectors.dense(vector.toArray)))
      } else {
        vectorWords.append(LabeledPoint(1d, Vectors.dense(vector.toArray)))
      }
    }
    PetuniaUtils.writeArray2File(vectorWords, "./libs/vector.txt")
    val data = sc.parallelize[LabeledPoint](vectorWords)
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Split data into training (70%) and test (30%).
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    val precision = metrics.precisionByThreshold.collect.toMap[Double, Double]
    val recall = metrics.recallByThreshold.collect.toMap[Double, Double]
    val fMeasure = metrics.fMeasureByThreshold.collect.toMap[Double, Double]
    println("Threshold\t\tPrecision\t\tRecall\t\tF-Measure")
    precision.foreach(x => {
      println(x._1 + "\t\t" + x._2 + "\t\t" + recall.get(x._1).get + "\t\t" + fMeasure.get(x._1).get)
    })

    println("Area under ROC = " + auROC)
  }
}