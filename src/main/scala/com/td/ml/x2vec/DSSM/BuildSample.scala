package com.td.ml.x2vec.DSSM

import com.td.ml.xdata.common.{UserHistory, Utils}

import scala.collection.mutable
import scala.util.Random

object BuildSample {

    final val ExcludedModules: Set[String] = Set("M029", "M030", "M040", "M041", "M042", "M043", "M045", "M046")

    def process(argMap: mutable.Map[String, String]): Unit = {

        val ss = Utils.createSparkSession
        //val ss = Utils.createLocalSparkSession

        val tsBound = argMap("ts_bound").toLong
        val nLastest = argMap("n_lastest").toInt // 每个用户只取最近的nLastest个行为，高频用户截断
        val dLastest = argMap("d_lastest").toInt // 每个行为序列保证在dLastest天内
        val itemFreqThres = argMap("item_freq_thres").toInt
        val seqMaxSize = argMap("seq_max_size").toInt
        val testFraction = argMap("test_fraction").toFloat
        val overSample = argMap("over_sample").toInt

        val SeqDayWindowMaxSize: Int = 24 * 3600 * 1000 * dLastest

        // 随机采样行为序列
        val rddRawSamples = ss.sparkContext.textFile(argMap("input_path_user_history")).flatMap {
            line =>
                val cols = line.split('\t')
                val diu = cols(0)
                val userHistory = new UserHistory().load(cols(1))
                val plays = userHistory.actions.filter {
                    event =>
                        !ExcludedModules.contains(event.module) &&
                          event.ts > tsBound &&
                          event.playTime.getOrElse[Int](0) >= 10
                }.sortBy(_.ts).takeRight(nLastest)

                Range.inclusive(5, plays.length - 1).flatMap {
                    labelIdx =>
                        val labelEvent = plays(labelIdx)
                        /*
                        val curOverSample = {
                            if (labelEvent.duration.get >= 120 || labelEvent.share || labelEvent.download) {
                                overSample * 5
                            } else if (labelEvent.duration.get >= 60 || labelEvent.fav || labelEvent.like) {
                                overSample * 3
                            } else {
                                overSample
                            }
                        }
                         */
                        val curOverSample = overSample
                        Range.inclusive(1, curOverSample).map {
                            _ =>
                                val predictNextN = Random.nextInt(2) + 1
                                val seqStartIdx: Int = math.max(
                                    Utils.inclusiveRandomInt(lower = 0, upper = labelIdx - predictNextN),
                                    labelIdx - seqMaxSize)
                                val seqStartTs = plays(labelIdx).ts - SeqDayWindowMaxSize
                                val sampledSeq = plays.slice(seqStartIdx, labelIdx)
                                  .filter(_.ts >= seqStartTs)
                                  .map(_.vid)
                                  .toArray
                                (diu, plays(labelIdx).vid, sampledSeq)
                        }.iterator
                }.iterator
        }.filter(_._3.nonEmpty).cache()

        // 生成视频vid字典
        val rddVocabVid = rddRawSamples.map {
            x => (x._2, 1)
        }.reduceByKey(_ + _)
          .filter(x => x._2 >= itemFreqThres)
          .sortBy(_._2, ascending = false, numPartitions = 1)
          .zipWithUniqueId()
          .cache()
        rddVocabVid.map(x => "%d\t%s\t%d".format(x._2, x._1._1, x._1._2)).saveAsTextFile(argMap("output_path_vocab_vid"))
        val localVocabVid = rddVocabVid.map(x => (x._1._1, x._2)).collect().toMap
        val vocabVidSize = localVocabVid.size
        val vocabVid = ss.sparkContext.broadcast(localVocabVid)

        // 根据字典转化样本
        val rddMappedSamples = rddRawSamples.map {
            case (diu, targetVid, playSeq) =>
                val playSeqFiltered = playSeq.filter(vocabVid.value.contains(_))
                if (playSeqFiltered.isEmpty || !vocabVid.value.contains(targetVid)) {
                    null
                } else {
                    val positiveId = vocabVid.value(targetVid)
                    // 播放序列ids
                    val playSeqIds = playSeqFiltered.map(x => vocabVid.value(x))

                    (
                      Utils.inclusiveRandomInt(0, 10000000),
                        "%s\t%d\t%s".format(
                            diu, positiveId,
                            playSeqIds.sortWith(_ < _).mkString(",")
                        )
                    )
                }
        }.filter(_ != null).sortByKey(ascending = true, numPartitions = 200)
          .map(_._2)

        if (testFraction >= 0.00001 && testFraction <= 0.99999) {
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
