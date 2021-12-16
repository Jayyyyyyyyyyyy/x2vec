package com.td.ml.x2vec.auxiliary

import com.td.ml.xdata.common.{Utils, VideoProfile}

import scala.collection.mutable

object ExtractVideoTitle {

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession

        val dateBound = argMap.getOrElse[String]("date_after", "2000-01-01")

        ss.sparkContext.textFile(argMap("input_path"))
          .map {
              line =>
                  val cols = line.split('\t')
                  val video = new VideoProfile
                  video.parse(cols(2))
                  if (video.isRecommendable && video.createTime >= dateBound && video.vid.nonEmpty && video.title.length >= 5 && video.title.length <= 30 &&
                    !video.title.contains("\r") && !video.title.contains("\n") && !video.title.contains("\t")) {
                      "%s\t%d\t%s".format(video.vid, video.updateTimestamp, video.title)
                  } else {
                      ""
                  }
          }.filter(_.nonEmpty)
          .repartition(5)
          .saveAsTextFile(argMap("output_path"))
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
