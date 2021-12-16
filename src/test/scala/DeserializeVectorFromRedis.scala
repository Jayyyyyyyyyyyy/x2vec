import redis.clients.jedis.Jedis

import scala.collection.mutable

object DeserializeVectorFromRedis {

    def main(args: Array[String]): Unit = {
        val jedis = new Jedis("10.42.27.232", 6379)

        val bytes = jedis.get("1500678984219".getBytes)
        if (bytes != null && bytes.length % 4 == 0) {
            val output: mutable.MutableList[Float] = mutable.MutableList()
            val dim = bytes.length / 4
            for (i <- 0 until dim) {
                val f = java.nio.ByteBuffer.wrap(bytes.slice(i * 4, i * 4 + 4)).order(java.nio.ByteOrder.LITTLE_ENDIAN).getFloat()
                // val f = java.lang.Float.intBitsToFloat(bytes(i * 4) ^ (bytes(i * 4 + 1) << 8) ^ (bytes(i * 4 + 2) << 16) ^ (bytes(i * 4 + 3) << 24))
                output += f
            }
            println(output.mkString(","))
        }

        jedis.close()
    }

}
