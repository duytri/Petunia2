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
    val conf = new SparkConf().setAppName("ISLab.Petunia")
    val sc = new SparkContext(conf)
    val broadcastArgs = sc.broadcast(args)

    //~~~~~~~~~~Get all data directories~~~~~~~~~~

    val inputDirPath = broadcastArgs.value(2) + "/data/in"

    val input = (inputDirPath + File.separator + "0", inputDirPath + File.separator + "1")

    //val inputDirFile0 = new File(input0)
    //val inputDirFile1 = new File(input1)

    //~~~~~~~~~~Get all input files~~~~~~~~~~
    val listFiles0 = sc.parallelize((new File(input._1)).listFiles)
    val listFiles1 = sc.parallelize((new File(input._2)).listFiles)

    //var wordSetByFile = new ArrayBuffer[Map[String, Int]](inputFiles.length) // Map[word, frequency in document]
    var wordSetByFile0: RDD[Map[String, Int]] = sc.emptyRDD[Map[String, Int]]
    //Foreach text file
    listFiles0.map { x =>
      wordSetByFile0 ++ sc.parallelize(PetuniaUtils.statWords(x))
    }

    var wordSetByFile1: RDD[Map[String, Int]] = sc.emptyRDD[Map[String, Int]]
    //Foreach text file
    listFiles1.map { x =>
      wordSetByFile1 ++ sc.parallelize(PetuniaUtils.statWords(x))
    }

    println("So file 0: " + listFiles0.count)
    println("So file 1: " + listFiles1.count)

    //~~~~~~~~~~Calculate TFIDF~~~~~~~~~~
    var tfidfWordSet0: RDD[Map[String, Double]] = sc.emptyRDD[Map[String, Double]] // Map[word, TF*IDF-value]
    wordSetByFile0.map(oneFile => {
      tfidfWordSet0 ++ sc.parallelize(PetuniaUtils.statTFIDF(oneFile, wordSetByFile0))
    })

    var tfidfWordSet1: RDD[Map[String, Double]] = sc.emptyRDD[Map[String, Double]] // Map[word, TF*IDF-value]
    wordSetByFile1.map(oneFile => {
      tfidfWordSet1 ++ sc.parallelize(PetuniaUtils.statTFIDF(oneFile, wordSetByFile1))
    })

    //~~~~~~~~~~Remove stopwords~~~~~~~~~~
    //// Load stopwords from file
    val stopwordFilePath = broadcastArgs.value(2) + "/libs/vietnamese-stopwords.txt"
    var arrStopwords = new ArrayBuffer[String]
    val swSource = Source.fromFile(stopwordFilePath)
    swSource.getLines.foreach { x => arrStopwords.append(x) }
    swSource.close
    //// Foreach document, remove stopwords
    tfidfWordSet0.foreach(oneFile => oneFile --= arrStopwords)
    tfidfWordSet1.foreach(oneFile => oneFile --= arrStopwords)

    //~~~~~~~~~~Normalize by TFIDF~~~~~~~~~~
    val lowerUpperBound = (broadcastArgs.value(0).toDouble, broadcastArgs.value(1).toDouble)
    println("Argument 0 (lower bound): " + lowerUpperBound._1 + " - Argument 1 (upper bound): " + lowerUpperBound._2)
    var attrWords: RDD[String] = sc.emptyRDD[String]
    tfidfWordSet0.foreach(oneFile => {
      attrWords ++ sc.parallelize(oneFile.filter(x => x._2 > lowerUpperBound._1 && x._2 < lowerUpperBound._2).keySet.toSeq)
    })
    tfidfWordSet1.foreach(oneFile => {
      attrWords ++ sc.parallelize(oneFile.filter(x => x._2 > lowerUpperBound._1 && x._2 < lowerUpperBound._2).keySet.toSeq)
    })

    //~~~~~~~~~~Create vector~~~~~~~~~~
    var vectorWords: RDD[LabeledPoint] = sc.emptyRDD[LabeledPoint]
    /*for (i <- 0 to inputFiles.length - 1) {
      var vector = new ArrayBuffer[Double]
      for (word <- attrWords) {
        if (tfidfWordSet(i).contains(word)) {
          vector.append(tfidfWordSet(i).get(word).get)
        } else vector.append(0d)
      }
      if (i < numFiles0) {
        vectorWords.append(LabeledPoint(0d, Vectors.dense(vector.toArray)))
      } else {
        vectorWords.append(LabeledPoint(1d, Vectors.dense(vector.toArray)))
      }
    }*/
    tfidfWordSet0.foreach(oneFile => {
      var vector = new ArrayBuffer[Double]
      attrWords
    })
    
    //PetuniaUtils.writeArray2File(vectorWords, "./libs/vector.txt")
    //val data = sc.parallelize[LabeledPoint](vectorWords)
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
