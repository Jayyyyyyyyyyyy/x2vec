#!/usr/bin/env bash

source /etc/profile

dt=$1

output_path=/user/wangshuang/apps/x2vec/auxiliary/i2i-eval-samples/dt=${dt}
${HADOOP_HOME}/bin/hadoop fs -test -e ${output_path}/_SUCCESS
if [ $? -eq 0 ]; then
    echo "output path ${output_path} found, skip"
    exit 0
fi

${HADOOP_HOME}/bin/hadoop fs -rm -r ${output_path}

${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --class com.td.ml.x2vec.base.I2IEvaluationSample \
    --name x2vec.i2i-eval-samples.${dt} \
    --driver-memory 2G \
    --executor-memory 1G \
    --executor-cores 2 \
    --num-executors 100 \
    --queue root.spark \
    --conf spark.executor.memoryOverhead=2048 \
    ../lib/x2vec-1.0-SNAPSHOT.jar \
    negatives=32 \
    sample_rate=0.2 \
    input_path_merge_log=/user/wangshuang/apps/xdata/merge_log/dt=${dt} \
    output_path=${output_path}