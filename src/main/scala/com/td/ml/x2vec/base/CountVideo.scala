package com.td.ml.x2vec.base

import com.td.ml.xdata.common.{UserHistory, Utils, VideoProfile, ModuleManager}

import scala.collection.mutable

object CountVideo {

    final val ExcludedModules: Set[String] = Set("M029", "M045")

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession

        val tsBound = argMap("ts_bound").toLong
        val clickThreshold = argMap("click_threshold").toInt
        val feedOnly = argMap("feed_only").toBoolean
        val starLevelThreshold = argMap("star_level_threshold").toInt
        val requireRecommendable = argMap("require_recommendable").toBoolean
        val bigVideoOnly = argMap("big_video_only").toBoolean

        val rddVideos = ss.sparkContext.textFile(argMap("input_path_user_history")).flatMap {
            line =>
                val cols = line.split('\t')
                val userHistory = new UserHistory().load(cols(1))
                userHistory.actions.filter {
                    x =>
                        !ExcludedModules.contains(x.module) &&
                            x.played &&
                            x.ts > tsBound &&
                            x.playTime.getOrElse[Int](0) > 5 &&
                            (!feedOnly || ModuleManager.isModuleFeed(x.module))
                }.map(x => (x.vid, 1))
                    .iterator
        }.reduceByKey(_ + _)
            .filter(_._2 >= clickThreshold)

        if (requireRecommendable || starLevelThreshold > 0 || bigVideoOnly) {
            ss.sparkContext.textFile(argMap("input_path_video_profile")).map {
                line =>
                    val profile = new VideoProfile
                    profile.parse(line)
                    val idValidVideo =
                        (!requireRecommendable || profile.isRecommendable) &&
                            (starLevelThreshold <= 0 || !profile.isDance || profile.userLevel >= starLevelThreshold) &&
                            (!bigVideoOnly || !profile.isSmallVideo)
                    (profile.vid, idValidVideo)
            }.filter(_._2)
                .join(rddVideos)
                .map(x => "%s\t%d".format(x._1, x._2._2))
                .repartition(10)
                .saveAsTextFile(argMap("output_path"))
        } else {
            rddVideos.map(x => "%s\t%d".format(x._1, x._2))
                .repartition(10)
                .saveAsTextFile(argMap("output_path"))
        }
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
