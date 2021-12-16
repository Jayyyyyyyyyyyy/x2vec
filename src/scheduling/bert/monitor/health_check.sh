#!/usr/bin/env bash

hostname=`hostname`
curdir=`pwd`
owner="wangshuang"

bert_streaming_process_cnt=`ps -ef | grep bert64.conf | grep realtime | grep -v "grep" | wc -l`
if [[ ${bert_streaming_process_cnt} -ne 1 ]]; then
    echo "bert_streaming_process_cnt=${bert_streaming_process_cnt}, status not ok"
    curl "http://10.10.77.161/sms.php?tos=18510411406&content=【bert64-streaming崩溃!!!】【host】${hostname}【path】${curdir}"
    curl -X POST -H "'Content-type':'application/json'" -d '{"msgtype":"markdown","markdown":{"content":"bert64-streaming崩溃\n >owner ：<font color=\"warning\">'${owner}'</font>\n >机器 ：<font color=\"warning\">'${hostname}'</font> \n > 路径：<font color=\"warning\">'${curdir}'</font>"}}' https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=ac428ee8-c6b6-48d7-ae61-649f8a9f6ea9
else
    echo "bert-streaming ok"
fi

