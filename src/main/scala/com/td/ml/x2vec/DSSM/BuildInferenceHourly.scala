package com.td.ml.x2vec.DSSM

import com.td.ml.xdata.common.{UserHistory, Utils}
import com.alibaba.fastjson.JSON
import scala.collection.mutable

object BuildInferenceHourly {

    def filterSeq(seq: Array[(Long, String)], num: Int): Array[Long] = {
        val distincts: mutable.Set[Long] = mutable.Set()
        val result: mutable.MutableList[Long] = mutable.MutableList()
        seq.sortWith((x, y) => x._2 > y._2).foreach {
            x =>
                if (!distincts.contains(x._1)) {
                    result += x._1
                    distincts.add(x._1)
                }
        }
        result.toArray.take(num)
    }

    def extract(line: String, vocab: Map[String, Long]): (String, (Long, String)) = {
        try {
            val json = JSON.parseObject(line)
            val diu = json.getString("u_diu")
            val playtime = Utils.string2Int(json.getString("u_playtime"))
            val timestamp = json.getString("u_timestamp")
            val vid = json.getString("u_vid")


            if (diu != null && diu.nonEmpty &&
                vid != null && vid.nonEmpty &&
                timestamp != null && timestamp.nonEmpty &&
                vocab.contains(vid) &&
                playtime.getOrElse[Int](0) >= 10) {
                return (diu, (vocab(vid), timestamp))
            }
        } catch {
            case exception: Exception =>
                System.err.println(exception.getMessage)
                System.err.println(line)
        }
        null
    }

    def process(argMap: mutable.Map[String, String]): Unit = {

        val ss = Utils.createSparkSession

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

        val rddHourlySamples = ss.sparkContext.textFile(argMap("input_path_video_play_speed")).map {
            line =>
                extract(line, vocabDictVid.value)
        }.filter(_ != null)
          .groupByKey()

        // 随机采样行为序列
        val rddHistorySamples = ss.sparkContext.textFile(argMap("input_path_user_history")).map {
            line =>
                val cols = line.split('\t')
                val diu = cols(0)
                val userHistory = new UserHistory().load(cols(1))
                val plays = userHistory.actions.filter {
                    event =>
                        event.ts > tsBound &&
                          event.playTime.getOrElse[Int](0) >= 10 &&
                          vocabDictVid.value.contains(event.vid)
                }.sortBy(_.ts)

                val seq = plays.filter(_.ts >= plays.last.ts - SeqDayWindowMaxSize)
                  .takeRight(seqMaxSize).map(x => vocabDictVid.value(x.vid)).toArray

                (diu, seq)
        }.filter(_._2.nonEmpty)

        rddHourlySamples.leftOuterJoin(rddHistorySamples).map {
            case (diu, (hourlySamples, historySamples)) =>
                val history = if (historySamples.isDefined) historySamples.get else Array[Long]()
                val hourly = filterSeq(hourlySamples.toArray, seqMaxSize)
                val finalSeq = hourly.union(history.takeRight(seqMaxSize - hourly.length))
                if (finalSeq.length < seqMinSize) {
                    ""
                } else {
                    "%s\t%s\t%s".format(
                        diu, "0", hourly.union(history.takeRight(seqMaxSize - hourly.length)).sortWith(_ < _).mkString(",")
                    )
                }
        }.filter(_.nonEmpty)
          .coalesce(100)
          .saveAsTextFile(argMap("output_path"))
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
