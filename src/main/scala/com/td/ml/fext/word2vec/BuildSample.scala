package com.td.ml.fext.word2vec

import scala.collection.mutable
import com.td.ml.xdata.common.{UserHistory, Utils, VideoProfile}

object BuildSample {

    def preprocessSequence(seq: List[String]): List[String] = {
        val newSeq: mutable.MutableList[String] = mutable.MutableList[String]()
        Range.inclusive(0, seq.length - 1).foreach {
            i =>
                if (i == 0 || seq(i) != seq(i - 1)) {
                    newSeq += seq(i)
                }
        }
        newSeq.toList
    }

    def process(argMap: mutable.Map[String, String]): Unit = {

        val ss = Utils.createSparkSession

        val itemFreqThres = argMap("item_freq_thres").toInt
        val tsBound = argMap("ts_bound").toLong
        val seqMinSize = argMap("seq_min_size").toLong

        val rddImpression = ss.sparkContext.textFile(argMap("input_path_user_history")).flatMap {
            line =>
                val cols = line.split('\t')
                val diu = cols(0)
                val history = new UserHistory().load(cols(1))
                history.actions.filter {
                    event =>
                        event.ts > tsBound &&
                            event.playTime.getOrElse[Int](0) >= 20
                }.map {
                    x =>
                        (x.vid, (diu, history.registerId, x.ts))
                }
        }

        val rddVideo = ss.sparkContext.textFile(argMap("input_path_video_profile")).map {
            line =>
                val cols = line.split('\t')
                if (cols.length != 3) {
                    null
                } else {
                    val video = new VideoProfile
                    video.parse(cols(2))
                    if (video.isSmallVideo ||
                      cols(0).isEmpty ||
                      (video.uid.isEmpty && video.mp3.name.isEmpty)) {
                        null
                    } else {
                        (cols(0), (video.uid, video.mp3.name))
                    }
                }
        }.filter(x => x != null).cache()

        val rddSequence = rddImpression.join(rddVideo)
          .filter {
              x => x._2._1._2.isEmpty || x._2._2._1.isEmpty || x._2._1._2 != x._2._2._1
          }.map {
              x =>
                  // (diu, (ts, uid, mp3, ...))
                  (x._2._1._1, (x._2._1._3, x._2._2._1, x._2._2._2))
          }.groupByKey().cache()

        // author_uid samples
        val rddAuthorFilter = rddSequence.flatMap(
            _._2.toList.map(_._2).distinct.filter(x => x.nonEmpty && !x.contains(" ")).map((_, 1)))
          .reduceByKey(_ + _)
          .filter(_._2 >= itemFreqThres)
          .map(_._1).cache()
        val authorFilter = ss.sparkContext.broadcast(rddAuthorFilter.collect().toSet)

        rddSequence.map {
            case (_, iter) =>
                val sentence = preprocessSequence(
                    iter.toList.sortWith((x, y) => x._1 < y._1)
                      .map(_._2)
                      .filter(x => authorFilter.value.contains(x)))
                if (sentence.size >= seqMinSize) {
                    sentence.mkString(" ")
                } else {
                    ""
                }
        }.filter(_.nonEmpty)
          .repartition(50)
          .saveAsTextFile(argMap("output_path_vauthor_sample"))

        // mp3 samples
        val rddMp3Filter = rddSequence.flatMap(
            _._2.toList.map(_._3).distinct.filter(x => x.nonEmpty && !x.contains(" ")).map((_, 1)))
          .reduceByKey(_ + _)
          .filter(_._2 >= itemFreqThres)
          .map(_._1).cache()
        val mp3Filter = ss.sparkContext.broadcast(rddMp3Filter.collect().toSet)

        rddSequence.map {
            case (_, iter) =>
                val sentence = preprocessSequence(
                    iter.toList.sortWith((x, y) => x._1 < y._1)
                      .map(_._3)
                      .filter(x => mp3Filter.value.contains(x)))
                if (sentence.size >= seqMinSize) {
                    sentence.mkString(" ")
                } else {
                    ""
                }
        }.filter(_.nonEmpty)
          .repartition(50)
          .saveAsTextFile(argMap("output_path_vmp3_sample"))
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
