package com.td.ml.x2vec.base

import com.redis._
import com.td.ml.xdata.common.Utils
import scala.collection.mutable

object FeedRedis {

    def floatToLittleEndianBytes(x: Float): Array[Byte] = {
        // little-endian
        val l = java.lang.Float.floatToIntBits(x)
        val a = Array.fill(4)(0.toByte)
        for (i <- 0 to 3) a(i) = ((l >> (i * 8)) & 0xff).toByte
        a
    }

    def packVectorIntoBinary(vector: Array[Float]): Array[Byte] = {
        val buffer = java.nio.ByteBuffer.allocate(4 * vector.length)
        vector.foreach(x => buffer.put(floatToLittleEndianBytes(x)))
        buffer.array()
    }

    def process(argMap: mutable.Map[String, String]): Unit = {
        val ss = Utils.createSparkSession

        val redisHost = argMap("redis").split(':')(0)
        val redisPort = argMap("redis").split(':')(1).toInt
        val dim = argMap("dim").toInt
        val ttl = argMap.getOrElse[String]("ttl", "-1").toInt

        val redis = new RedisClient(redisHost, redisPort)
        if (argMap("flushall").toBoolean) {
            if (!redis.flushall) {
                throw new RuntimeException("flushall failed")
            }
        }
        if (argMap.contains("version")) {
            val version = argMap("version").trim
            if (!redis.set("version", version)) {
                throw new RuntimeException("set version failed")
            }
        }

        ss.sparkContext.textFile(argMap("input_path"))
          .repartition(argMap("concurrency").toInt)
          .foreachPartition {
              lines =>
                  val redis = new RedisClient(redisHost, redisPort)
                  lines.foreach {
                      line =>
                          val cols = line.split('\t')
                          val key = cols(0).trim
                          val vector = cols(1).split(',').map(x => x.toFloat)
                          if (key.nonEmpty && vector.length == dim) {
                              if (ttl <= 0) {
                                  redis.set(key, packVectorIntoBinary(vector))
                              } else {
                                  redis.setex(key, ttl, packVectorIntoBinary(vector))
                              }
                          }
                  }
          }
    }

    def main(args: Array[String]): Unit = {
        val argMap = Utils.parseArgs(args)
        process(argMap)
    }

}
