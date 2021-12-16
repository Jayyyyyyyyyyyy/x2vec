package com.td.ml.x2vec.bert

import com.alibaba.fastjson.{JSON => FastJSON}
import com.td.ml.xdata.common.Utils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable

object MergeEmbedding {

    def prepareEmbedding(ss: SparkSession, input: String): RDD[(String, (String, String))] = {
        ss.sparkContext.textFile(input).map {
            line =>
                val json = FastJSON.parseObject(line)
                if (!json.containsKey("time")) {
                    null
                } else {
                    val vid = json.getString("vid")
                    val t = json.getString("time")
                    (vid, (line, t))
                }
        }.filter(_ != null)
    }

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession

        val rddIncoming = prepareEmbedding(ss, argMap("input_path_incoming_embedding"))

        if (argMap.contains("input_path_embedding_history")) {
            val rddHistory = prepareEmbedding(ss, argMap("input_path_embedding_history"))
            rddHistory.union(rddIncoming)
              .reduceByKey((x, y) => if (x._2 >= y._2) x else y)
              .map(_._2._1)
              .repartition(500)
              .saveAsTextFile(argMap("output_path"), classOf[org.apache.hadoop.io.compress.GzipCodec])
        } else {
            rddIncoming.reduceByKey((x, y) => if (x._2 >= y._2) x else y)
              .map(_._2._1)
              .repartition(500)
              .saveAsTextFile(argMap("output_path"), classOf[org.apache.hadoop.io.compress.GzipCodec])
        }
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
