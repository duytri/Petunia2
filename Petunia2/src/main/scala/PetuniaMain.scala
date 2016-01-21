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
import java.util.Properties
import java.io.File
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
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
    var inputFiles = inputDirFile0.listFiles ++ inputDirFile1.listFiles

    var wordSetByFile = new Array[HashMap[String, Int]](inputFiles.length) // Map[word, frequency in document]
    //Foreach text file
    for (i <- 0 to inputFiles.length-1) {
      var wordsTmpArr = new ArrayBuffer[String]
      Source.fromFile(inputFiles(i).getAbsolutePath, "utf-8").getLines().foreach { x => wordsTmpArr.append(x) }
      PetuniaUtils.addOrIgnore(wordSetByFile(i), wordsTmpArr)
    }
    //~~~~~~~~~~Calculate TFIDF~~~~~~~~~~
    var tfidfWordSet = new Array[HashMap[String, Double]](inputFiles.length) // Map[word, TF*IDF-value]
    for (i <- 0 to inputFiles.length - 1) {
      for (oneWord <- wordSetByFile(i)) {
        tfidfWordSet(i) += oneWord._1 -> TFIDFCalc.tfIdf(oneWord, i, wordSetByFile)
      }
    }

    //~~~~~~~~~~Remove stopwords~~~~~~~~~~
    //// Load stopwords from file
    val stopwordFilePath = "./libs/vietnamese-stopwords.txt"
    var arrStopwords = new ArrayBuffer[String]
    Source.fromFile(stopwordFilePath, "utf-8").getLines().foreach { x => arrStopwords.append(x) }
    //// Foreach document, remove stopwords
    for (i <- 0 to inputFiles.length - 1) {
      tfidfWordSet(i) --= arrStopwords
    }

    //~~~~~~~~~~Normalize by TFIDF~~~~~~~~~~
    val lowerUpperBound = (0, 1)
    var attrWords = ArrayBuffer[String]()
    for (i <- 0 to inputFiles.length - 1) {
      tfidfWordSet(i).foreach(x => {
        if (x._2 <= lowerUpperBound._1 || x._2 >= lowerUpperBound._2) {
          tfidfWordSet(i).remove(x._1)
        } else attrWords += x._1 
      })
    }

    //~~~~~~~~~~Create vector~~~~~~~~~~
    var vectorWords: RDD[LabeledPoint] = sc.emptyRDD[LabeledPoint]
    val rdd: RDD[LabeledPoint] = sc.emptyRDD[LabeledPoint]
    for (i <- 0 to inputFiles.length - 1) {
      var vector = new ArrayBuffer[Double]
      for (word <- attrWords) {
        if (tfidfWordSet(i).contains(word)) {
          vector.append(tfidfWordSet(i).get(word).get)
        } else vector.append(0d)
      }
      if (i < inputDirFile0.length)
        vectorWords ++ PetuniaUtils.convert2RDD(rdd, LabeledPoint(0, Vectors.dense(vector.toArray)))
      else
        vectorWords ++ PetuniaUtils.convert2RDD(rdd, LabeledPoint(1, Vectors.dense(vector.toArray)))
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Split data into training (70%) and test (30%).
    val splits = vectorWords.randomSplit(Array(0.7, 0.3), seed = 11L)
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

    println("Area under ROC = " + auROC)
  }
}