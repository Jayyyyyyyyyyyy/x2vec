package com.td.ml.x2vec.GraphEmbedding

import com.td.ml.xdata.common.{UserHistory, Utils}

import scala.collection.mutable
import scala.util.Random

object DeepWalkWeightedOnlyVideo {

    final val ExcludedModules: Set[String] = Set("M029", "M030", "M040", "M041", "M042", "M043", "M045", "M046")

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession
        val tsBound = argMap("ts_bound").toLong
        val seqWindowSize = argMap("seq_window_size").toInt
        val vertexFreqThres = argMap("vertex_freq_thres").toInt
        val edgeFreqThres = argMap("edge_freq_thres").toInt
        val adjacencyMaxSize = argMap("adjacency_max_size").toInt
        val walkLength = argMap("walk_length").toInt
        val walkMinLength = argMap("walk_min_length").toInt
        val seqDistinctItems = argMap("seq_distinct_items").toInt
        val walksPerVertex = argMap("walks_per_vertex").toInt

        // 每个用户内部生成item序列
        val rddRawSeqs = ss.sparkContext.textFile(argMap("input_path_user_history")).map {
            line =>
                val cols = line.split('\t')
                val userHistory = new UserHistory().load(cols(1))

                userHistory.actions.filter {
                    event =>
                        !ExcludedModules.contains(event.module) &&
                          event.ts > tsBound &&
                          event.playTime.getOrElse[Int](0) >= 5
                }.sortBy(_.ts).map(_.vid).toArray
        }.cache()

        // 生成字典
        val rddVocab = rddRawSeqs.flatMap(_.iterator)
          .map((_, 1))
          .reduceByKey(_ + _)
          .filter(_._2 >= vertexFreqThres)
          .sortBy(_._2, ascending = false, numPartitions = 1)
          .map(x => x._1)
          .cache()
        rddVocab.saveAsTextFile(argMap("output_path_vocab"))
        val vocabDict = ss.sparkContext.broadcast(rddVocab.collect().toSet)

        val rddRawWalks = rddRawSeqs.map {
            x => x.filter(vocabDict.value.contains(_))
        }.filter {
            x =>
                x.length >= walkMinLength && x.distinct.length >= seqDistinctItems
        }.map(_.mkString(" "))

        if (walksPerVertex > 0) {

            // 生成graph的连接矩阵
            val rddAdjacency = rddRawSeqs.flatMap {
                seq =>
                    val edges: mutable.MutableList[((String, String), Int)] = mutable.MutableList()

                    if (seq.length >= 5) {
                        for (center <- seq.indices) {
                            if (vocabDict.value.contains(seq(center))) {
                                Range.inclusive(1, seqWindowSize).foreach {
                                    margin =>
                                        val neighbor = center + margin
                                        if (margin != 0 &&
                                          neighbor >= 0 &&
                                          neighbor < seq.length &&
                                          seq(center) != seq(neighbor) &&
                                          vocabDict.value.contains(seq(neighbor))) {
                                            edges += (((seq(center), seq(neighbor)), 1))
                                        }
                                }
                            }
                        }
                    }
                    edges.iterator

            }.reduceByKey(_ + _)
              .filter(_._2 >= edgeFreqThres)
              .map(x => (x._1._1, (x._1._2, x._2)))
              .groupByKey()
              .map {
                  case (vertex, neighbors) =>
                      var weightedNeighbors: Array[(String, Float)] = neighbors.toArray
                        .sortWith((x, y) => x._2 > y._2)
                        .take(adjacencyMaxSize)
                        .map(x => (x._1, math.sqrt(x._2.toDouble).toFloat))

                      val videoAvgCnt = math.round(weightedNeighbors.map(_._2).sum * 1.0f / weightedNeighbors.length)
                      var currentBorder: Int = 0
                      weightedNeighbors = weightedNeighbors.map {
                          x =>
                              currentBorder = currentBorder + math.round(10 * math.min(x._2, videoAvgCnt))
                              (x._1, currentBorder.toFloat)
                      }

                      (vertex, weightedNeighbors)

              }.cache()

            rddAdjacency.map {
                case (vertex, weightedNeighbors) =>
                    "%s\t%s".format(vertex, weightedNeighbors.map(x => "%s|%.1f".format(x._1, x._2)).mkString(" "))
            }.saveAsTextFile(argMap("output_path_adjacency_matrix"))

            // 随机游走
            var keyedRandomWalks = rddAdjacency.keys.flatMap {
                node =>
                    Range.inclusive(1, walksPerVertex).map {
                        _ =>
                            val walks = new Array[String](walkLength)
                            walks(0) = node
                            (node, (walks, 1))
                    }
            }

            for (iter <- 1 until walkLength) {
                val grownRandomWalks =
                    rddAdjacency.rightOuterJoin(keyedRandomWalks)
                      .map {
                          case (vertex, (neighbours, (walks, distance))) =>
                              if (neighbours.isDefined) {
                                  val randomNum = Random.nextInt(neighbours.get.last._2.toInt)
                                  val pickIdx = neighbours.get.indexWhere(_._2 > randomNum)
                                  walks(distance) = neighbours.get(pickIdx)._1
                                  (walks(distance), (walks, distance + 1))
                              } else {
                                  (vertex, (walks, distance))
                              }
                      }

                keyedRandomWalks.unpersist()
                keyedRandomWalks = grownRandomWalks
            }

            keyedRandomWalks.values
              .filter {
                  x =>
                      x._2 >= walkMinLength && x._1.distinct.length >= seqDistinctItems
              }.map(x => x._1.take(x._2).mkString(" "))
              .union(rddRawWalks)
              .saveAsTextFile(argMap("output_path_deep_walk"))
        } else {
            rddRawWalks.saveAsTextFile(argMap("output_path_deep_walk"))
        }
    }


    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }
}
