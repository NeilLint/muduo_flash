#!/bin/bash

# 启动程序
./muduo data/stories110M.bin data/tokenizer.bin data/input_prompt.txt &
pid=$!

max_total_rss=0

while kill -0 $pid 2>/dev/null; do
    # 统计所有进程RSS总和（单位KB）
    total_rss=$(awk '{ sum+=$6 } END { print sum }' /proc/*/statm 2>/dev/null)

    # 也可以用 ps 命令，注意格式可能不同，这里用 ps 格式方便
    # total_rss=$(ps -eo rss= | awk '{sum+=$1} END {print sum}')

    if [[ $total_rss -gt $max_total_rss ]]; then
        max_total_rss=$total_rss
    fi
    sleep 0.5
done

echo "Max total system RSS: ${max_total_rss} KB"
