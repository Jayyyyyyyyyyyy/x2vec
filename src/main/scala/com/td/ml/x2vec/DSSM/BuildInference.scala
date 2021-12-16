package com.td.ml.x2vec.DSSM

import com.td.ml.xdata.common.{UserHistory, Utils}
import scala.collection.mutable

object BuildInference {

    final val ExcludedModules: Set[String] = Set("M029", "M045")

    def process(argMap: mutable.Map[String, String]): Unit = {

        val ss = Utils.createSparkSession
        //val ss = Utils.createLocalSparkSession

        val tsBound = argMap("ts_bound").toLong
        val dLastest = argMap("d_lastest").toInt // 每个行为序列保证在dLastest天内
        val seqMaxSize = argMap("seq_max_size").toInt
        val seqMinSize = argMap("seq_min_size").toInt

        val SeqDayWindowMaxSize: Int = 24 * 3600 * 1000 * dLastest

        val localVocabVid = ss.sparkContext.textFile(argMap("input_path_vocab_vid")).map {
            line =>
                val cols = line.split('\t')
                (cols(1), cols(0).toLong)
        }.collect().toMap
        val vocabDictVid = ss.sparkContext.broadcast(localVocabVid)

        // 随机采样行为序列
        val rddRawSamples = ss.sparkContext.textFile(argMap("input_path_user_history")).map {
            line =>
                val cols = line.split('\t')
                val diu = cols(0)
                val userHistory = new UserHistory().load(cols(1))
                val plays = userHistory.actions.filter {
                    event =>
                        !ExcludedModules.contains(event.module) &&
                          event.ts > tsBound &&
                          event.playTime.getOrElse[Int](0) >= 10 &&
                          vocabDictVid.value.contains(event.vid)
                }.sortBy(_.ts)

                val playSeqIds = plays.filter(_.ts >= plays.last.ts - SeqDayWindowMaxSize)
                  .takeRight(seqMaxSize).map(x => vocabDictVid.value(x.vid)).sortWith(_ < _)
                if (playSeqIds.isEmpty && playSeqIds.length < seqMinSize) {
                    ""
                } else {
                    "%s\t%s\t%s".format(
                        diu, "0", playSeqIds.mkString(",")
                    )
                }
        }.filter(_.nonEmpty)

        rddRawSamples.saveAsTextFile(argMap("output_path"))
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
