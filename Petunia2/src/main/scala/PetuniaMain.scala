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
    println("Start...")
    //~~~~~~~~~~Initialization~~~~~~~~~~
    val conf = new SparkConf().setAppName("ISLab.Petunia")
    val sc = new SparkContext(conf)
    val broadcastArgs = sc.broadcast(args)

    println("Finish initializing configuration! Getting all input...")

    //~~~~~~~~~~Get all data directories~~~~~~~~~~

    val inputDirPath = broadcastArgs.value(2) + "input" + File.separator + "in"

    val input = (inputDirPath + File.separator + "0", inputDirPath + File.separator + "1")

    //val inputDirFile0 = new File(input0)
    //val inputDirFile1 = new File(input1)

    //~~~~~~~~~~Get all input files~~~~~~~~~~
    val listFiles0 = sc.parallelize(PetuniaUtils.getListOfSubFiles((new File(input._1))))
    val listFiles1 = sc.parallelize(PetuniaUtils.getListOfSubFiles((new File(input._2))))

    //var wordSetByFile = new ArrayBuffer[Map[String, Int]](inputFiles.length) // Map[word, frequency in document]
    var wordSetByFile0: RDD[Map[String, Int]] = sc.emptyRDD[Map[String, Int]]
    //Foreach text file
    wordSetByFile0 = wordSetByFile0.union(listFiles0.map { fileDir =>
      PetuniaUtils.statWords(fileDir)
    })

    var wordSetByFile1: RDD[Map[String, Int]] = sc.emptyRDD[Map[String, Int]]
    //Foreach text file
    wordSetByFile1 = wordSetByFile1.union(listFiles1.map { fileDir =>
      PetuniaUtils.statWords(fileDir)
    })

    println("Finish collecting and distributing input! Start calculate TFIDF...")

    //~~~~~~~~~~Calculate TFIDF~~~~~~~~~~
    var tfidfWordSet0: RDD[Map[String, Double]] = sc.emptyRDD[Map[String, Double]] // Map[word, TF*IDF-value]
    var wordSet0 = wordSetByFile0.collect
    var wordSet1 = wordSetByFile1.collect
    val bcWordSet0 = sc.broadcast(wordSet0)
    val bcWordSet1 = sc.broadcast(wordSet1)
    tfidfWordSet0 = tfidfWordSet0.union(wordSetByFile0.map(oneFile => {
      PetuniaUtils.statTFIDF(oneFile, bcWordSet0.value)
    }))

    var tfidfWordSet1: RDD[Map[String, Double]] = sc.emptyRDD[Map[String, Double]] // Map[word, TF*IDF-value]
    tfidfWordSet1 = tfidfWordSet1.union(wordSetByFile1.map(oneFile => {
      PetuniaUtils.statTFIDF(oneFile, bcWordSet1.value)
    }))
    tfidfWordSet0.cache
    tfidfWordSet1.cache
    println("Finish caching TFIDF-word-set! Removing stopwords...")

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

    println("Removed stopwords! Start normalizing TFIDF-word-set...")

    //~~~~~~~~~~Normalize by TFIDF~~~~~~~~~~
    val lowerUpperBound = (broadcastArgs.value(0).toDouble, broadcastArgs.value(1).toDouble)
    println("Argument 0 (lower bound): " + lowerUpperBound._1 + " - Argument 1 (upper bound): " + lowerUpperBound._2)
    var attrWords: RDD[String] = sc.emptyRDD[String]
    attrWords = attrWords.union(tfidfWordSet0.flatMap(oneFile => {
      oneFile.filter(x => x._2 > lowerUpperBound._1 && x._2 < lowerUpperBound._2).keySet
    }))
    attrWords = attrWords.union(tfidfWordSet1.flatMap(oneFile => {
      oneFile.filter(x => x._2 > lowerUpperBound._1 && x._2 < lowerUpperBound._2).keySet
    }))

    println("Let's create vectors...")

    //~~~~~~~~~~Create vector~~~~~~~~~~
    var vectorWords: RDD[LabeledPoint] = sc.emptyRDD[LabeledPoint]
    val arrAttribute = attrWords.collect
    val bcAttrWords = sc.broadcast(arrAttribute)
    vectorWords = vectorWords.union(tfidfWordSet0.map(oneFile => {
      var vector = new ArrayBuffer[Double]
      bcAttrWords.value.foreach { word =>
        {
          if (oneFile.contains(word)) {
            vector.append(oneFile.get(word).get)
          } else vector.append(0d)
        }
      }
      LabeledPoint(0d, Vectors.dense(vector.toArray))
    }))
    vectorWords = vectorWords.union(tfidfWordSet1.map(oneFile => {
      var vector = new ArrayBuffer[Double]
      bcAttrWords.value.foreach { word =>
        {
          if (oneFile.contains(word)) {
            vector.append(oneFile.get(word).get)
          } else vector.append(0d)
        }
      }
      LabeledPoint(1d, Vectors.dense(vector.toArray))
    }))

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println("And now the time to start SVM model training...")

    println("Split data into training (70%) and test (30%).")
    val splits = vectorWords.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    println("Run training algorithm to build the model.")
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    println("Compute raw scores on the test set.")
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    println("Get evaluation metrics.")
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    //val auROC = metrics.areaUnderROC()

    val auPR = metrics.areaUnderPR
    println("Area under PR-Curve = " + auPR)

    val precision = metrics.precisionByThreshold.collect.toMap[Double, Double]
    val recall = metrics.recallByThreshold.collect.toMap[Double, Double]
    val fMeasure = metrics.fMeasureByThreshold.collect.toMap[Double, Double]
    println("Threshold\t\tPrecision\t\tRecall\t\tF-Measure")
    precision.foreach(x => {
      println(x._1 + "\t\t" + x._2 + "\t\t" + recall.get(x._1).get + "\t\t" + fMeasure.get(x._1).get)
    })

    /*println("Score\t\t\tPoint")
    scoreAndLabels.collect.foreach(x => {
      println(x._1 + "\t\t\t" + x._2)
    })*/

    println("Save and load model.")
    model.save(sc, broadcastArgs.value(3)) //save in "myDir+data" may cause error
    val sameModel = SVMModel.load(sc, broadcastArgs.value(3))

    println("Testing...")
    val testWords = PetuniaUtils.statWords(broadcastArgs.value(4)) //Load word set
    var tfidfTest = Map[String, Double]()
    //Calculate TFIDF
    tfidfTest ++= testWords.filter(wordSet0.contains).map(oneWord => {
      val tf = oneWord._2 / testWords.foldLeft(0d)(_ + _._2)
      val idf = Math.log10(wordSet0.size / wordSet0.filter(x => { x.contains(oneWord._1) }).length)
      oneWord._1 -> tf * idf
    })
    tfidfTest ++= testWords.filter(wordSet1.contains).map(oneWord => {
      val tf = oneWord._2 / testWords.foldLeft(0d)(_ + _._2)
      val idf = Math.log10(wordSet1.size / wordSet1.filter(x => { x.contains(oneWord._1) }).length)
      oneWord._1 -> tf * idf
    })
    //Remove stopwords
    tfidfTest --= arrStopwords
    //Create vector
    var testVector = new ArrayBuffer[Double]
    arrAttribute.foreach { word =>
      {
        if (tfidfTest.contains(word)) {
          testVector.append(tfidfTest.get(word).get)
        } else testVector.append(0d)
      }
    }
    //Test
    val result = sameModel.predict(Vectors.dense(testVector.toArray))
    
  }
}
