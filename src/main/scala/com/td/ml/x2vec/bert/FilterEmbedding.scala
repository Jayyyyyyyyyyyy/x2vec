package com.td.ml.x2vec.bert

import com.td.ml.xdata.common.Utils
import com.alibaba.fastjson.{JSON => FastJSON}
import scala.collection.mutable

object FilterEmbedding {

    def normalize(array: Array[Double]): Array[Double] = {
        val base = math.sqrt(array.map(x => x * x).sum)
        array.map(x => x / base)
    }

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession

        val performNormalize = argMap.getOrElse[String]("normalize", "false").toBoolean
        val ctrBound = argMap("ctr_bound").toDouble
        val displayBound = argMap("display_bound").toInt
        val clickBound = argMap("click_bound").toInt

        val videoFilters = ss.sparkContext.broadcast(
            ss.sparkContext.textFile(argMap("input_path_video_ctr_stat")).map {
                line =>
                    val cols = line.split('\t')
                    val vid = cols(0)
                    val ctr = cols(1).toDouble
                    val display = cols(2).toInt
                    val click = cols(3).toInt
                    if (ctr >= ctrBound && display >= displayBound && click >= clickBound) {
                        vid
                    } else {
                        ""
                    }
            }.filter(_.nonEmpty)
                .collect.toSet
        )

        ss.sparkContext.textFile(argMap("input_path_bert_embedding")).map {
            line =>
                val json = FastJSON.parseObject(line)
                val vid = json.getString("vid")
                val emb = json.getString("embed")

                if (vid != null && videoFilters.value.contains(vid) && emb != null && emb.nonEmpty) {
                    if (performNormalize) {
                        val normEmb = normalize(emb.split(',').map(_.toDouble)).map(x => "%.4f".format(x)).mkString(",")
                        "%s\t%s".format(vid, normEmb)
                    } else {
                        "%s\t%s".format(vid, emb)
                    }
                } else {
                    null
                }
        }.filter(_ != null).repartition(20).saveAsTextFile(argMap("output_path"))

    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
