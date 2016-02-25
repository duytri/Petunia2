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
      wordSetByFile0 = wordSetByFile0.union(sc.parallelize(PetuniaUtils.statWords(x)))
    }

    var wordSetByFile1: RDD[Map[String, Int]] = sc.emptyRDD[Map[String, Int]]
    //Foreach text file
    listFiles1.map { x =>
      wordSetByFile1 = wordSetByFile1.union(sc.parallelize(PetuniaUtils.statWords(x)))
    }

    println("So file 0: " + listFiles0.count)
    println("So file 1: " + listFiles1.count)

    //~~~~~~~~~~Calculate TFIDF~~~~~~~~~~
    var tfidfWordSet0: RDD[Map[String, Double]] = sc.emptyRDD[Map[String, Double]] // Map[word, TF*IDF-value]
    val bcWordSet0 = sc.broadcast(wordSetByFile0.collect)
    val bcWordSet1 = sc.broadcast(wordSetByFile1.collect)
    wordSetByFile0.map(oneFile => {
      tfidfWordSet0 = tfidfWordSet0.union(sc.parallelize(PetuniaUtils.statTFIDF(oneFile, bcWordSet0.value)))
    })

    var tfidfWordSet1: RDD[Map[String, Double]] = sc.emptyRDD[Map[String, Double]] // Map[word, TF*IDF-value]
    wordSetByFile1.map(oneFile => {
      tfidfWordSet1 = tfidfWordSet1.union(sc.parallelize(PetuniaUtils.statTFIDF(oneFile, bcWordSet1.value)))
    })
    tfidfWordSet0.cache
    tfidfWordSet1.cache

    //~~~~~~~~~~Remove stopwords~~~~~~~~~~
    //// Load stopwords from file
    val stopwordFilePath = broadcastArgs.value(2) + "/libs/vietnamese-stopwords.txt"
    var arrStopwords = new ArrayBuffer[String]
    val swSource = Source.fromFile(stopwordFilePath)
    swSource.getLines.foreach { x => arrStopwords.append(x) }
    swSource.close
    val bcStopwords = sc.broadcast(arrStopwords)
    //// Foreach document, remove stopwords
    tfidfWordSet0.foreach(oneFile => oneFile --= bcStopwords.value)
    tfidfWordSet1.foreach(oneFile => oneFile --= bcStopwords.value)

    //~~~~~~~~~~Normalize by TFIDF~~~~~~~~~~
    val lowerUpperBound = (broadcastArgs.value(0).toDouble, broadcastArgs.value(1).toDouble)
    println("Argument 0 (lower bound): " + lowerUpperBound._1 + " - Argument 1 (upper bound): " + lowerUpperBound._2)
    var attrWords: RDD[String] = sc.emptyRDD[String]
    tfidfWordSet0.foreach(oneFile => {
      attrWords = attrWords.union(sc.parallelize(oneFile.filter(x => x._2 > lowerUpperBound._1 && x._2 < lowerUpperBound._2).keySet.toSeq))
    })
    tfidfWordSet1.foreach(oneFile => {
      attrWords = attrWords.union(sc.parallelize(oneFile.filter(x => x._2 > lowerUpperBound._1 && x._2 < lowerUpperBound._2).keySet.toSeq))
    })

    //~~~~~~~~~~Create vector~~~~~~~~~~
    var vectorWords: RDD[LabeledPoint] = sc.emptyRDD[LabeledPoint]
    val bcAttrWords = sc.broadcast(attrWords.collect)
    tfidfWordSet0.foreach(oneFile => {
      var vector = new ArrayBuffer[Double]
      bcAttrWords.value.foreach { word =>
        {
          if (oneFile.contains(word)) {
            vector.append(oneFile.get(word).get)
          } else vector.append(0d)
        }
      }
      vectorWords = vectorWords.union(sc.parallelize(Seq(LabeledPoint(0d, Vectors.dense(vector.toArray)))))
    })
    tfidfWordSet1.foreach(oneFile => {
      var vector = new ArrayBuffer[Double]
      bcAttrWords.value.foreach { word =>
        {
          if (oneFile.contains(word)) {
            vector.append(oneFile.get(word).get)
          } else vector.append(0d)
        }
      }
      vectorWords = vectorWords.union(sc.parallelize(Seq(LabeledPoint(1d, Vectors.dense(vector.toArray)))))
    })

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
    /*val precision = metrics.precisionByThreshold.collect.toMap[Double, Double]
    val recall = metrics.recallByThreshold.collect.toMap[Double, Double]
    val fMeasure = metrics.fMeasureByThreshold.collect.toMap[Double, Double]
    println("Threshold\t\tPrecision\t\tRecall\t\tF-Measure")
    precision.foreach(x => {
      println(x._1 + "\t\t" + x._2 + "\t\t" + recall.get(x._1).get + "\t\t" + fMeasure.get(x._1).get)
    })*/
    println("Score\t\t\tPoint")
    scoreAndLabels.collect.foreach(x => {
      println(x._1 + "\t\t\t" + x._2)
    })

    println("Area under ROC = " + auROC)

    // Save and load model
    //model.save(sc, broadcastArgs.value(3))
    //val sameModel = SVMModel.load(sc, broadcastArgs.value(3))
  }
}
