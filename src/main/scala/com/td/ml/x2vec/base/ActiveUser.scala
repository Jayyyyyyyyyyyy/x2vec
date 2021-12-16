package com.td.ml.x2vec.base

import com.td.ml.xdata.common.{Utils, MergeLog}
import scala.collection.mutable

object ActiveUser {

    final val ExcludedModules: Set[String] = Set("M029", "M045")

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession
        val daysThres = argMap("days_thres").toInt
        val clickThres = argMap("clicks_thres").toInt

        ss.sparkContext.textFile(argMap("input_path")).map {
            line =>
                var dt: String = ""
                var plays: Int = 0
                val cols = line.split('\t')
                val mergelog = new MergeLog()
                mergelog.load(cols(1))

                mergelog.actions.filter(x => !ExcludedModules.contains(x.module)).foreach {
                    x =>
                        if (dt.isEmpty) {
                            dt = Utils.tsToDate(x.ts)
                        }
                        if (x.played || x.isDownload > 0 || x.isFav > 0) {
                            plays += 1
                        }
                }
                (cols(0), (dt, plays))
        }.groupByKey().map {
            row =>
                var totalClick: Int = 0
                val dates: mutable.Set[String] = mutable.Set[String]()
                row._2.foreach {
                    col =>
                        if (col._1.nonEmpty) {
                            dates.add(col._1)
                        }
                        totalClick += col._2
                }
                (row._1, dates.size, totalClick)
        }.filter(x => x._2 >= daysThres && x._3 >= clickThres)
          .map(_._1).repartition(5).saveAsTextFile(argMap("output_path"))
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
