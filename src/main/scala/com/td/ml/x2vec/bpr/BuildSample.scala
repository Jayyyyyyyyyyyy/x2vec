package com.td.ml.x2vec.bpr

import scala.collection.mutable
import com.td.ml.xdata.common.{UserHistory, Utils}

object BuildSample {

    final val ExcludedModules: Set[String] = Set("M029", "M030", "M040", "M041", "M042", "M043", "M045", "M046")

    def process(argMap: mutable.Map[String, String]): Unit = {

        val ss = Utils.createSparkSession

        val itemFreqThres = argMap("item_freq_thres").toInt
        val tsBound = argMap("ts_bound").toLong
        val nlastest = argMap("nlastest").toInt
        val testFraction = argMap("test_fraction").toDouble
        val userConfidence = argMap("user_confidence").toInt

        val rddRawSamples = ss.sparkContext.textFile(argMap("input_path_user_history")).flatMap {
            line =>
                val cols = line.split('\t')
                val diu = cols(0)
                val history = new UserHistory().load(cols(1))
                if (history.actions.size < userConfidence) {
                    Array[(String, String)]().iterator
                } else {
                    history.actions.filter {
                        event =>
                            !ExcludedModules.contains(event.module) &&
                              event.ts > tsBound &&
                              event.playTime.getOrElse[Int](0) >= 10
                    }.sortBy(_.ts)
                      .takeRight(nlastest)
                      .map(x => (diu, x.vid))
                      .iterator
                }
        }.repartition(200).cache()

        val rddItemIdMapping = rddRawSamples.map(x => (x._2, 1))
            .reduceByKey(_ + _)
            .filter(x => x._2 >= itemFreqThres)
            .sortBy[Int](_._2, ascending = false, numPartitions = 1)
            .map(_._1)
            .zipWithUniqueId
            .cache()

        val rddUserIdMapping = rddRawSamples.map(_._1).distinct.zipWithUniqueId.cache()

        rddItemIdMapping.map(x => "%s\t%d".format(x._1, x._2)).saveAsTextFile(argMap("output_path_item_ids"))
        rddUserIdMapping.map(x => "%s\t%d".format(x._1, x._2))
          .coalesce(20).saveAsTextFile(argMap("output_path_user_ids"))

        val rddMappedSamples = rddRawSamples.join(rddUserIdMapping).map {
            case (_, (vid, diuId)) =>
                (vid, diuId)
        }.join(rddItemIdMapping).map {
            case (_, (diuId, vidId)) =>
                // for random shuffling
                val hash = List.fill(16)(scala.util.Random.nextPrintableChar).mkString
                (hash, "%d %d".format(diuId, vidId))
        }.sortByKey(ascending = true, numPartitions = 200)
          .map(x => "%s 1".format(x._2))


        if (testFraction >= 0.0001 && testFraction <= 0.9999) {
            val rddSplits = rddMappedSamples.randomSplit(Array[Double](testFraction, 1 - testFraction))
            rddSplits(0).saveAsTextFile(argMap("output_path_test_samples"))
            rddSplits(1).saveAsTextFile(argMap("output_path_train_samples"))
        } else {
            rddMappedSamples.saveAsTextFile(argMap("output_path_train_samples"))
        }
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
