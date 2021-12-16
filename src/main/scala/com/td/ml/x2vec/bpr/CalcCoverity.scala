package com.td.ml.x2vec.bpr

import scala.collection.mutable
import com.td.ml.xdata.common.Utils

object CalcCoverity {

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession

        val rddUserIds = ss.sparkContext.textFile(argMap("input_path_user_ids")).map {
            line =>
                (line.split('\t')(0), 1)
        }.cache()

        val rddDv = ss.sparkContext.textFile(argMap("input_path_app_video_dv")).map {
            line =>
                val cols = line.split('\u0001')
                (cols(0), 1)
        }.cache()

        val rddDvCoverity = rddDv.leftOuterJoin(rddUserIds).map {
            case (_, (_, modelFlag)) =>
                if (modelFlag.isDefined) {
                    ("dv", (1, 1))
                } else {
                    ("dv", (1, 0))
                }
        }.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))

        val rddUvCoverity = rddDv.distinct.leftOuterJoin(rddUserIds).map {
            case (_, (_, modelFlag)) =>
                if (modelFlag.isDefined) {
                    ("uv", (1, 1))
                } else {
                    ("uv", (1, 0))
                }
        }.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))

        rddDvCoverity.union(rddUvCoverity).coalesce(1).saveAsTextFile(argMap("output_path_coverity"))
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}

